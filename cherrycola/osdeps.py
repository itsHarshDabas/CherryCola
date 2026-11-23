"""What differs between Linux, macOS and Windows: folders, processes, memory.

Everything else in CherryColaLLM is plain Python and HTTP. The XDG_* variables
win on every system (the tests rely on that).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

WINDOWS = sys.platform == "win32"


def _xdg(var: str, unix_default: str, windows_var: str) -> Path:
    if os.environ.get(var):
        return Path(os.environ[var])
    if WINDOWS:
        base = os.environ.get(windows_var) or os.path.expanduser("~")
        return Path(base) / "CherryCola"
    return Path(os.path.expanduser(unix_default))


def config_home() -> Path:
    """Holds grace/cherrycola.conf and grace/cherrycola-models.json."""
    return _xdg("XDG_CONFIG_HOME", "~/.config", "APPDATA")


def data_home() -> Path:
    return _xdg("XDG_DATA_HOME", "~/.local/share", "LOCALAPPDATA")


def state_home() -> Path:
    return _xdg("XDG_STATE_HOME", "~/.local/state", "LOCALAPPDATA")


def runtime_dir() -> Path:
    """Pid files: $XDG_RUNTIME_DIR/cherrycola, else the state folder."""
    if os.environ.get("XDG_RUNTIME_DIR"):
        return Path(os.environ["XDG_RUNTIME_DIR"]) / "cherrycola"
    return state_home() / "cherrycola"


# ---- processes ----------------------------------------------------------------

def popen_detached() -> dict:
    """Popen arguments so a server outlives the command that started it."""
    if WINDOWS:
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP
                 | getattr(subprocess, "DETACHED_PROCESS", 0x8)
                 | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
        return {"creationflags": flags, "close_fds": True}
    return {"start_new_session": True, "close_fds": True}


def _win_handle(pid: int, access: int):
    import ctypes
    return ctypes.windll.kernel32.OpenProcess(access, False, pid)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if WINDOWS:
        # Never os.kill(pid, 0) here: on Windows that sends Ctrl+C.
        import ctypes
        handle = _win_handle(pid, 0x1000)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle,
                                                       ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def process_name(pid: int) -> str | None:
    """The program's name (no .exe), or None when it can't be read."""
    if WINDOWS:
        import ctypes
        handle = _win_handle(pid, 0x1000)
        if not handle:
            return None
        buf = ctypes.create_unicode_buffer(1024)
        size = ctypes.c_ulong(len(buf))
        ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(
            handle, 0, buf, ctypes.byref(size))
        ctypes.windll.kernel32.CloseHandle(handle)
        if not ok:
            return None
        name = Path(buf.value).name
        return name[:-4] if name.lower().endswith(".exe") else name
    try:
        return Path(f"/proc/{pid}/comm").read_text("utf-8").strip()
    except OSError:
        pass
    if shutil.which("ps"):  # macOS and BSDs have no /proc
        try:
            out = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                                 capture_output=True, text=True, timeout=5)
            name = out.stdout.strip()
            return Path(name).name if name else None
        except (OSError, subprocess.SubprocessError):
            return None
    return None


def terminate(pid: int, force: bool = False) -> None:
    """Ask a process to stop (force: kill it). Raises OSError."""
    if WINDOWS:
        import ctypes
        handle = _win_handle(pid, 0x0001)  # PROCESS_TERMINATE
        if not handle:
            raise OSError(f"no process {pid}")
        try:
            if not ctypes.windll.kernel32.TerminateProcess(handle, 1):
                raise OSError(f"could not stop process {pid}")
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
        return
    os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)


# ---- memory ----------------------------------------------------------------------

def _meminfo(key: str, path: str = "/proc/meminfo") -> int | None:
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _win_memory() -> tuple[int, int] | None:
    import ctypes

    class MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MemStatus()
    status.dwLength = ctypes.sizeof(MemStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullTotalPhys, status.ullAvailPhys


def _sysctl(name: str) -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True,
                             text=True, timeout=5)
        return int(out.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def total_ram(meminfo: str = "/proc/meminfo") -> int | None:
    if WINDOWS:
        mem = _win_memory()
        return mem[0] if mem else None
    value = _meminfo("MemTotal", meminfo)
    if value is None and sys.platform == "darwin":
        value = _sysctl("hw.memsize")
    return value


def available_ram() -> int | None:
    if WINDOWS:
        mem = _win_memory()
        return mem[1] if mem else None
    return _meminfo("MemAvailable")


def link_file(src: Path, dest: Path) -> None:
    """A symlink; on Windows without symlink rights, a hard link."""
    try:
        dest.symlink_to(src)
    except OSError:
        if not WINDOWS:
            raise
        os.link(src, dest)  # same drive only; raises otherwise
