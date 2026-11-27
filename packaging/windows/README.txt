CherryColaLLM for Windows: the app (cherrycolallm-gui.exe, "CherryColaLLM" in the
Start menu) and the command line (cherrycolallm.exe)

1. Run cherrycolallm-<version>-windows-x64-setup.exe (no admin needed; it adds
   itself to your PATH, open a new terminal afterwards). Or unzip the .zip
   and put cherrycolallm.exe on your PATH, or "scoop install cherrycolallm".
   The old name cherrycola works too.
2. llama.cpp comes with it (in the llama.cpp folder next to cherrycolallm.exe):
   a Vulkan build that uses NVIDIA, AMD or Intel graphics through the normal
   driver and falls back to the CPU. Nothing else is needed to run a .gguf
   file: in the app, Models > File… loads it in place (no download, no copy).
   Optional engines: Ollama for Windows (winget install Ollama.Ollama), LM
   Studio with its local server on, or your own llama.cpp (say a CUDA build) on
   PATH, which wins over the bundled one.
3. Try it:
     cherrycolallm pull --use C:\path\to\model.gguf
     cherrycolallm recommend
     cherrycolallm ask "hello"

Settings live in %APPDATA%\CherryCola\grace\, models, chats and logs in
%LOCALAPPDATA%\CherryCola\. Everything in the README works the same way,
except the chat app, which needs GTK 4 and libadwaita (Linux for now).
