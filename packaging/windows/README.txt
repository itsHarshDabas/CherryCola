CherryColaLLM for Windows: the app (cherrycolallm-gui.exe, "CherryColaLLM" in the
Start menu) and the command line (cherrycolallm.exe)

1. Run cherrycolallm-<version>-windows-x64-setup.exe (no admin needed; it adds
   itself to your PATH, open a new terminal afterwards). Or unzip the .zip
   and put cherrycolallm.exe on your PATH, or "scoop install cherrycolallm".
   The old name cherrycola works too.
2. Install an engine:
   - Ollama for Windows (winget install Ollama.Ollama), the easiest; or
   - llama.cpp (winget install llama.cpp, or unzip a Windows release so llama-server.exe and
     llama-quantize.exe are on your PATH); or
   - LM Studio, with its local server turned on.
3. Try it:
     cherrycolallm recommend
     cherrycolallm pull            (Ollama)
     cherrycolallm ask "hello"

Settings live in %APPDATA%\CherryCola\grace\, models, chats and logs in
%LOCALAPPDATA%\CherryCola\. Everything in the README works the same way,
except the chat app, which needs GTK 4 and libadwaita (Linux for now).
