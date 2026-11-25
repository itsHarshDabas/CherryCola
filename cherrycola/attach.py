"""Files you can hand to a model: documents, images, audio and video.

Stdlib only. What a file turns into:

* text, code, CSV, JSON, ...      -> its text, in the prompt
* .docx .pptx .xlsx .odt .odp     -> their text (read straight from the zip)
* .pdf                            -> text via ``pdftotext`` (poppler) if present
* images                          -> sent to the model as images (needs a
                                     model that can see, e.g. gemma3, llava)
* video                           -> a handful of frames through ``ffmpeg``,
                                     sent as images; the soundtrack is
                                     transcribed too when a transcriber exists
* audio                           -> a transcript from ``whisper`` (OpenAI's
                                     CLI) or whisper.cpp's ``whisper-cli``
                                     (with ``whisper_model`` in the config)

Anything that needs a helper program says which one when it's missing.
"""

from __future__ import annotations

import base64
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

from . import osdeps

__all__ = ["AttachError", "Attachment", "load", "kind_of"]

MAX_TEXT_CHARS = 60_000
MAX_IMAGE_BYTES = 20 * 1024 * 1024
VIDEO_FRAMES = 6

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
OFFICE_EXT = {".docx", ".pptx", ".xlsx", ".odt", ".odp", ".ods"}


class AttachError(Exception):
    pass


@dataclass
class Attachment:
    text: str = ""  # goes into the prompt
    images: list[str] = field(default_factory=list)  # base64, no prefix


def kind_of(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"
    if ext == ".pdf":
        return "pdf"
    if ext in OFFICE_EXT:
        return "office"
    return "text"


def _clip(text: str, name: str) -> str:
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + f"\n[... {name} cut at {MAX_TEXT_CHARS} characters]"
    return text


def _block(name: str, body: str) -> str:
    return f"[Attached file: {name}]\n{_clip(body.strip(), name)}\n[End of {name}]\n\n"


def _text_file(path: Path) -> str:
    raw = path.read_bytes()[: MAX_TEXT_CHARS * 4]
    if b"\x00" in raw[:4096]:
        raise AttachError(f"{path.name} looks like a binary file I can't read "
                          "(images, documents, audio and video are supported)")
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _xml_text(data: bytes, para_tags: tuple[str, ...]) -> str:
    out: list[str] = []
    root = ElementTree.fromstring(data)
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag in ("t", "p", "h") and el.text and tag == "t":
            out.append(el.text)
        elif tag == "text" and el.text:
            out.append(el.text)
        if tag in para_tags:
            out.append("\n")
    return "".join(out)


def _office(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if ext == ".docx":
                return _xml_text(z.read("word/document.xml"), ("p",))
            if ext == ".pptx":
                slides = sorted((n for n in names
                                 if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                                key=lambda n: int(re.findall(r"\d+", n)[-1]))
                return "\n\n".join(f"--- slide {i} ---\n" + _xml_text(z.read(n), ("p",))
                                   for i, n in enumerate(slides, 1))
            if ext == ".xlsx":
                strings: list[str] = []
                if "xl/sharedStrings.xml" in names:
                    root = ElementTree.fromstring(z.read("xl/sharedStrings.xml"))
                    for si in root:
                        strings.append("".join(t.text or "" for t in si.iter()
                                               if t.tag.endswith("}t")))
                sheets = sorted(n for n in names
                                if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
                out = []
                for n in sheets:
                    out.append(f"--- {Path(n).stem} ---")
                    for row in ElementTree.fromstring(z.read(n)).iter():
                        if not row.tag.endswith("}row"):
                            continue
                        cells = []
                        for c in row:
                            v = next((x.text for x in c if x.tag.endswith("}v")), None)
                            if v is None:
                                inline = "".join(t.text or "" for t in c.iter()
                                                 if t.tag.endswith("}t"))
                                cells.append(inline)
                            elif c.get("t") == "s" and v.isdigit() and int(v) < len(strings):
                                cells.append(strings[int(v)])
                            else:
                                cells.append(v)
                        out.append("\t".join(cells))
                return "\n".join(out)
            return _xml_text(z.read("content.xml"), ("p", "h"))  # OpenDocument
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise AttachError(f"could not read {path.name}: {exc}") from exc


def _pdf(path: Path) -> str:
    tool = osdeps.which("pdftotext")
    if not tool:
        raise AttachError(f"reading {path.name} needs pdftotext (poppler): "
                          "winget install poppler, or save the PDF as text")
    try:
        out = subprocess.run([tool, "-layout", "-q", str(path), "-"],
                             capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise AttachError(f"pdftotext failed on {path.name}: {exc}") from exc
    if out.returncode != 0:
        raise AttachError(f"pdftotext could not read {path.name}")
    return out.stdout.decode("utf-8", errors="replace")


def _image(path: Path) -> str:
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise AttachError(f"{path.name} is bigger than "
                          f"{MAX_IMAGE_BYTES // 2**20} MB")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _transcribe(path: Path, model: str = "") -> str:
    """A transcript, or an AttachError that names what to install."""
    with tempfile.TemporaryDirectory() as tmp:
        cpp = osdeps.which("whisper-cli") or osdeps.which("whisper-cpp")
        if cpp:
            if not model:
                raise AttachError("set whisper_model in cherrycola.conf to a "
                                  "whisper.cpp ggml model file")
            wav = path
            ffmpeg = osdeps.which("ffmpeg")
            if path.suffix.lower() != ".wav" and ffmpeg:
                wav = Path(tmp) / "in.wav"
                subprocess.run([ffmpeg, "-y", "-i", str(path), "-ar", "16000",
                                "-ac", "1", str(wav)], capture_output=True,
                               timeout=600)
            run = subprocess.run([cpp, "-m", model, "-f", str(wav), "-nt"],
                                 capture_output=True, text=True, timeout=1800)
            if run.returncode == 0 and run.stdout.strip():
                return run.stdout.strip()
            raise AttachError(f"whisper-cli failed: {run.stderr.strip()[-200:]}")
        py = osdeps.which("whisper")
        if py:
            subprocess.run([py, str(path), "--output_format", "txt",
                            "--output_dir", tmp], capture_output=True,
                           timeout=3600)
            out = Path(tmp) / (path.stem + ".txt")
            if out.is_file():
                return out.read_text("utf-8", errors="replace").strip()
            raise AttachError("whisper could not transcribe " + path.name)
    raise AttachError(f"turning {path.name} into text needs a transcriber: "
                      "pip install openai-whisper, or whisper.cpp "
                      "(whisper-cli plus whisper_model in the config)")


def _video(path: Path, whisper_model: str) -> Attachment:
    ffmpeg = osdeps.which("ffmpeg")
    if not ffmpeg:
        raise AttachError(f"reading {path.name} needs ffmpeg: "
                          "winget install ffmpeg")
    att = Attachment()
    with tempfile.TemporaryDirectory() as tmp:
        probe = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True,
                               text=True, timeout=60)
        m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", probe.stderr)
        seconds = (int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])) if m else 60
        every = max(1.0, seconds / VIDEO_FRAMES)
        subprocess.run([ffmpeg, "-y", "-i", str(path), "-vf",
                        f"fps=1/{every:.2f},scale=768:-2", "-frames:v",
                        str(VIDEO_FRAMES), str(Path(tmp) / "f%02d.jpg")],
                       capture_output=True, timeout=300)
        frames = sorted(Path(tmp).glob("f*.jpg"))
        if not frames:
            raise AttachError(f"could not read frames from {path.name}")
        att.images = [_image(f) for f in frames]
        note = f"{len(frames)} frames from the video, evenly spaced"
        try:
            audio = Path(tmp) / "a.wav"
            subprocess.run([ffmpeg, "-y", "-i", str(path), "-vn", "-ar",
                            "16000", "-ac", "1", str(audio)],
                           capture_output=True, timeout=600)
            if audio.is_file() and audio.stat().st_size > 4096:
                note += ".\nSoundtrack transcript:\n" + _transcribe(audio, whisper_model)
        except AttachError as exc:
            note += f" (no transcript: {exc})"
        att.text = _block(path.name, note)
    return att


def load(paths: list[str], whisper_model: str = "") -> Attachment:
    """Everything the model should see for these files."""
    total = Attachment()
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_file():
            raise AttachError(f"no such file: {raw}")
        kind = kind_of(path)
        try:
            if kind == "image":
                total.images.append(_image(path))
                total.text += f"[Attached image: {path.name}]\n\n"
            elif kind == "video":
                one = _video(path, whisper_model)
                total.text += one.text
                total.images += one.images
            elif kind == "audio":
                total.text += _block(path.name, _transcribe(path, whisper_model))
            elif kind == "pdf":
                total.text += _block(path.name, _pdf(path))
            elif kind == "office":
                total.text += _block(path.name, _office(path))
            else:
                total.text += _block(path.name, _text_file(path))
        except OSError as exc:
            raise AttachError(f"could not read {path.name}: {exc}") from exc
    return total
