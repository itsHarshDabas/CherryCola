Name:           cherrycolallm
Version:        @VERSION@
Release:        1
Summary:        Run AI models on your own PC (Ollama, llama.cpp, LM Studio)
License:        LicenseRef-unspecified
URL:            https://github.com/itsHarshDabas/CherryColaLLM
Provides:       cherrycola = %{version}-%{release}
Obsoletes:      cherrycola < 0.3.0
BuildArch:      noarch
Requires:       python3 >= 3.12
Recommends:     python3-gobject
Recommends:     gtk4
Recommends:     libadwaita

%description
CherryColaLLM is a command line tool and a GTK chat app for local models. It
drives Ollama, llama.cpp's llama-server or LM Studio's server, with
per-model load and sampling settings, every llama.cpp option, requantizing,
and streaming models bigger than RAM from disk. Install Ollama or llama.cpp
separately; nothing is downloaded until you ask for a model.

%install
cp -a %{_sourcedir}/root/. %{buildroot}/

%files
/usr/bin/cherrycolallm
/usr/bin/cherrycolallm-chat
/usr/bin/cherrycola
/usr/bin/cherrycola-chat
/usr/lib/cherrycola
/usr/share/applications/org.graceos.CherryCola.desktop
/usr/share/icons/hicolor/scalable/apps/org.graceos.CherryCola.svg
%doc /usr/share/doc/cherrycolallm/README.md
