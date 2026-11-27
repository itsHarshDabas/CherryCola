; Inno Setup script: builds cherrycolallm-<version>-windows-x64-setup.exe.
; Per-user install (no admin prompt), adds the folder to the user's PATH.
; Build:  iscc /DAppVersion=1.0 /DSourceExe=dist\cherrycolallm.exe /DGuiExe=gui\target\release\cherrycolallm-gui.exe /DLlamaDir=dist\llama.cpp packaging\windows\cherrycolallm.iss
; LlamaDir is llama.cpp from packaging/fetch-llamacpp.sh: it goes in {app}\llama.cpp,
; where cherrycolallm.exe looks for it (osdeps.bundled).
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef GuiExe
  #define GuiExe "..\..\gui\target\release\cherrycolallm-gui.exe"
#endif
#ifndef LlamaDir
  #define LlamaDir "..\..\dist\llama.cpp"
#endif
#ifndef SourceExe
  #define SourceExe "..\..\dist\cherrycolallm.exe"
#endif

[Setup]
AppId={{6F0D2A1C-3B7E-4C55-9A52-0C4E8D7B1A36}
AppName=CherryColaLLM
AppVersion={#AppVersion}
AppPublisher=Harsh Dabas
AppPublisherURL=https://github.com/itsHarshDabas/CherryColaLLM
DefaultDirName={localappdata}\Programs\CherryColaLLM
PrivilegesRequired=lowest
DisableProgramGroupPage=yes

SetupIconFile=..\..\gui\assets\icon.ico
UninstallDisplayIcon={app}\cherrycolallm-gui.exe
OutputBaseFilename=cherrycolallm-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesEnvironment=yes
UninstallDisplayName=CherryColaLLM

[Files]
Source: "{#SourceExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.txt"; DestDir: "{app}"; Flags: ignoreversion isreadme
Source: "{#GuiExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#LlamaDir}\*"; DestDir: "{app}\llama.cpp"; Flags: ignoreversion recursesubdirs
; The old command name still works.
Source: "{#SourceExe}"; DestDir: "{app}"; DestName: "cherrycola.exe"; Flags: ignoreversion

[InstallDelete]
; Drop the files of an older bundled llama.cpp before copying the new one.
Type: filesandordirs; Name: "{app}\llama.cpp"

[Icons]
Name: "{autoprograms}\CherryColaLLM"; Filename: "{app}\cherrycolallm-gui.exe"
Name: "{autodesktop}\CherryColaLLM"; Filename: "{app}\cherrycolallm-gui.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Run]
Filename: "{app}\cherrycolallm-gui.exe"; Description: "Open CherryColaLLM"; Flags: nowait postinstall skipifsilent

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
  ValueData: "{olddata};{app}"; Check: NeedsAddPath(ExpandConstant('{app}'))

[Code]
function NeedsAddPath(Param: string): Boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Lowercase(Param) + ';', ';' + Lowercase(OrigPath) + ';') = 0;
end;
