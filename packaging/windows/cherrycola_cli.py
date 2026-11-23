"""Entry point for the Windows build (PyInstaller makes cherrycolallm.exe)."""

from cherrycola.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
