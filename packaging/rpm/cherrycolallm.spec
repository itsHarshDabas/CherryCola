Name:           cherrycolallm
Version:        @VERSION@
Release:        1
Summary:        Run AI models on your own PC (llama.cpp built in, Ollama, LM Studio)
%global debug_package %{nil}
%global __strip /bin/true
%{!?cc_arch: %global cc_arch x86_64}
License:        LicenseRef-unspecified
URL:            https://github.com/itsHarshDabas/CherryColaLLM
Provides:       cherrycola = %{version}-%{release}
Obsoletes:      cherrycola < 0.3.0
BuildArch:      %{cc_arch}
Requires:       python3 >= 3.12
# The bundled llama.cpp needs these; its own libraries stay private.
AutoReqProv:    no
Requires:       vulkan-loader
Requires:       libgomp
Recommends:     python3-gobject
Recommends:     gtk4
Recommends:     libadwaita

%description
CherryColaLLM is a command line tool and a GTK chat app for local models. It
drives Ollama, llama.cpp's llama-server or LM Studio's server, with
per-model load and sampling settings, every llama.cpp option, requantizing,
and streaming models bigger than RAM from disk. llama.cpp (a Vulkan build
that also runs on the CPU) comes with it, so .gguf files load right away;
Ollama and LM Studio are optional. Nothing is downloaded until you ask.

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
%license /usr/share/licenses/cherrycolallm
