#!/usr/bin/env bash
# Build the Linux packages into dist/: a .tar.gz of the install tree, a
# .deb (dpkg-deb) and an .rpm (rpmbuild), all x86_64 with llama.cpp bundled
# in /usr/lib/cherrycola/llama.cpp (packaging/fetch-llamacpp.sh). Arch
# packages come from packaging/arch/PKGBUILD (built with makepkg on Arch,
# see release.yml).
# Usage: packaging/build-linux.sh [VERSION]   (default: pyproject version)
# NO_LLAMACPP=1 builds without llama.cpp (no download).
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=${1:-$(python3 -c 'import tomllib;print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')}
DIST=$PWD/dist
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$DIST"
LLAMA=$STAGE/llama.cpp
if [ -z "${NO_LLAMACPP:-}" ]; then
  packaging/fetch-llamacpp.sh linux "$LLAMA"
fi

# The same layout GraceOS's package uses.
install_tree() {
  local root=$1
  install -d "$root/usr/lib/cherrycola"
  cp -r cherrycola "$root/usr/lib/cherrycola/"
  find "$root/usr/lib/cherrycola" -name __pycache__ -prune -exec rm -rf {} +
  install -Dm755 bin/cherrycolallm "$root/usr/bin/cherrycolallm"
  install -Dm755 bin/cherrycolallm-chat "$root/usr/bin/cherrycolallm-chat"
  # The old names (CherryCola until 0.3.0) stay as links.
  ln -sf cherrycolallm "$root/usr/bin/cherrycola"
  ln -sf cherrycolallm-chat "$root/usr/bin/cherrycola-chat"
  install -Dm644 data/org.graceos.CherryCola.desktop \
    "$root/usr/share/applications/org.graceos.CherryCola.desktop"
  install -Dm644 data/org.graceos.CherryCola.svg \
    "$root/usr/share/icons/hicolor/scalable/apps/org.graceos.CherryCola.svg"
  install -Dm644 README.md "$root/usr/share/doc/cherrycolallm/README.md"
  install -d "$root/usr/share/licenses/cherrycolallm"
  # The bundled llama.cpp: osdeps.bundled() finds it next to the package.
  if [ -d "$LLAMA" ]; then
    cp -a "$LLAMA" "$root/usr/lib/cherrycola/llama.cpp"
    install -Dm644 "$LLAMA/LICENSE" "$root/usr/share/licenses/cherrycolallm/llama.cpp-LICENSE"
  fi
}

# tar.gz
install_tree "$STAGE/tar/cherrycolallm-$VERSION"
tar -C "$STAGE/tar" -czf "$DIST/cherrycolallm-$VERSION-linux-x86_64.tar.gz" "cherrycolallm-$VERSION"

# .deb
DEB=$STAGE/deb
install_tree "$DEB"
install -d "$DEB/DEBIAN"
sed "s/@VERSION@/$VERSION/" packaging/debian/control > "$DEB/DEBIAN/control"
if [ ! -d "$LLAMA" ]; then
  sed -i 's/^Architecture: .*/Architecture: all/; s/^Depends: .*/Depends: python3 (>= 3.12)/' "$DEB/DEBIAN/control"
fi
echo "Installed-Size: $(du -sk "$DEB/usr" | cut -f1)" >> "$DEB/DEBIAN/control"
ARCH=$(sed -n 's/^Architecture: //p' "$DEB/DEBIAN/control")
dpkg-deb --root-owner-group --build "$DEB" "$DIST/cherrycolallm_${VERSION}_${ARCH}.deb" >/dev/null

# .rpm
if command -v rpmbuild >/dev/null; then
  RPM=$STAGE/rpm
  mkdir -p "$RPM"/{SOURCES,SPECS,BUILD,RPMS,SRPMS}
  install_tree "$RPM/SOURCES/root"
  sed "s/@VERSION@/$VERSION/" packaging/rpm/cherrycolallm.spec > "$RPM/SPECS/cherrycolallm.spec"
  RPMARCH=x86_64; [ -d "$LLAMA" ] || RPMARCH=noarch
  rpmbuild --quiet -bb --define "_topdir $RPM" --target "$RPMARCH" \
    --define "cc_arch $RPMARCH" "$RPM/SPECS/cherrycolallm.spec"
  cp "$RPM"/RPMS/*/*.rpm "$DIST/"
else
  echo "rpmbuild not found: skipping the .rpm" >&2
fi
ls -1 "$DIST"
