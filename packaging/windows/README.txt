CherryColaLLM for Windows (command line)

1. Put cherrycolallm.exe somewhere on your PATH (or run it from this folder).
   Installed with Scoop, the old name cherrycola works too.
2. Install an engine:
   - Ollama for Windows (https://ollama.com), the easiest; or
   - llama.cpp: unzip a Windows release so llama-server.exe and
     llama-quantize.exe are on your PATH; or
   - LM Studio, with its local server turned on.
3. Try it:
     cherrycolallm recommend
     cherrycolallm pull            (Ollama)
     cherrycolallm ask "hello"

Settings live in %APPDATA%\CherryCola\grace\, models, chats and logs in
%LOCALAPPDATA%\CherryCola\. Everything in the README works the same way,
except the chat app, which needs GTK 4 and libadwaita (Linux for now).
