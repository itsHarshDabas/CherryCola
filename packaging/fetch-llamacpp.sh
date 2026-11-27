#!/usr/bin/env bash
# Download the pinned llama.cpp build (packaging/llamacpp.env), check its
# sha256, and copy the parts CherryColaLLM uses into DEST: llama-server,
# llama-quantize, llama-imatrix, their libraries and the licence.
# Usage: packaging/fetch-llamacpp.sh linux|windows DEST
set -euo pipefail
cd "$(dirname "$0")/.."
OS=${1:?linux or windows}
DEST=${2:?destination folder}
# shellcheck disable=SC1091
. packaging/llamacpp.env
case $OS in
  linux)   FILE=llama-$LLAMACPP_TAG-bin-ubuntu-vulkan-x64.tar.gz; SUM=$LLAMACPP_LINUX_SHA256 ;;
  windows) FILE=llama-$LLAMACPP_TAG-bin-win-vulkan-x64.zip;      SUM=$LLAMACPP_WINDOWS_SHA256 ;;
  *) echo "unknown OS $OS" >&2; exit 1 ;;
esac
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
curl -fsSL --retry 4 -o "$WORK/$FILE" \
  "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMACPP_TAG/$FILE"
echo "$SUM  $WORK/$FILE" | sha256sum -c -
mkdir -p "$WORK/x" "$DEST"
if [ "$OS" = linux ]; then
  tar -xzf "$WORK/$FILE" -C "$WORK/x"
else
  if command -v unzip >/dev/null; then
    unzip -q "$WORK/$FILE" -d "$WORK/x"
  else  # Git Bash on Windows may lack unzip
    powershell -NoProfile -Command "Expand-Archive -LiteralPath '$(cygpath -w "$WORK/$FILE")' -DestinationPath '$(cygpath -w "$WORK/x")'"
  fi
fi
SRC=$(dirname "$(find "$WORK/x" -name 'llama-server*' -type f | head -n1)")
for p in llama-server llama-quantize llama-imatrix; do
  if [ "$OS" = windows ]; then
    cp "$SRC/$p.exe" "$DEST/"
    # The .exe files are small launchers for <name>-impl.dll.
    [ -f "$SRC/$p-impl.dll" ] && cp "$SRC/$p-impl.dll" "$DEST/"
  else
    cp -P "$SRC/$p" "$DEST/"
    for lib in "$SRC"/lib"$p"-impl.so*; do [ -e "$lib" ] && cp -P "$lib" "$DEST/"; done
  fi
done
if [ "$OS" = windows ]; then
  # Every library except the other tools' -impl ones.
  for f in "$SRC"/*.dll; do
    case $(basename "$f") in llama-*-impl.dll) ;; *) cp "$f" "$DEST/" ;; esac
  done
else
  for f in "$SRC"/lib*.so*; do
    case $(basename "$f") in libllama-*-impl.so*) ;; *) cp -P "$f" "$DEST/" ;; esac
  done
fi
for f in "$SRC"/LICENSE*; do [ -e "$f" ] && cp "$f" "$DEST/"; done
# llama.cpp's own MIT licence (the Windows zip only carries OpenMP's).
[ -f "$DEST/LICENSE" ] || curl -fsSL --retry 4 -o "$DEST/LICENSE" \
  "https://raw.githubusercontent.com/ggml-org/llama.cpp/$LLAMACPP_TAG/LICENSE"
echo "llama.cpp $LLAMACPP_TAG ($OS) -> $DEST: $(du -sh "$DEST" | cut -f1)"
