#!/usr/bin/env bash
# Build the Linux packages into dist/: a .tar.gz of the install tree, a
# .deb (dpkg-deb) and an .rpm (rpmbuild). Arch packages come from
# packaging/arch/PKGBUILD (built with makepkg on Arch, see release.yml).
# Usage: packaging/build-linux.sh [VERSION]   (default: pyproject version)
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=${1:-$(python3 -c 'import tomllib;print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')}
DIST=$PWD/dist
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$DIST"

# The same layout GraceOS's package uses.
install_tree() {
  local root=$1
  install -d "$root/usr/lib/cherrycola"
  cp -r cherrycola "$root/usr/lib/cherrycola/"
  find "$root/usr/lib/cherrycola" -name __pycache__ -prune -exec rm -rf {} +
  install -Dm755 bin/cherrycola "$root/usr/bin/cherrycola"
  install -Dm755 bin/cherrycola-chat "$root/usr/bin/cherrycola-chat"
  install -Dm644 data/org.graceos.CherryCola.desktop \
    "$root/usr/share/applications/org.graceos.CherryCola.desktop"
  install -Dm644 data/org.graceos.CherryCola.svg \
    "$root/usr/share/icons/hicolor/scalable/apps/org.graceos.CherryCola.svg"
  install -Dm644 README.md "$root/usr/share/doc/cherrycola/README.md"
}

# tar.gz
install_tree "$STAGE/tar/cherrycola-$VERSION"
tar -C "$STAGE/tar" -czf "$DIST/cherrycola-$VERSION-linux.tar.gz" "cherrycola-$VERSION"

# .deb
DEB=$STAGE/deb
install_tree "$DEB"
install -d "$DEB/DEBIAN"
sed "s/@VERSION@/$VERSION/" packaging/debian/control > "$DEB/DEBIAN/control"
echo "Installed-Size: $(du -sk "$DEB/usr" | cut -f1)" >> "$DEB/DEBIAN/control"
dpkg-deb --root-owner-group --build "$DEB" "$DIST/cherrycola_${VERSION}_all.deb" >/dev/null

# .rpm
if command -v rpmbuild >/dev/null; then
  RPM=$STAGE/rpm
  mkdir -p "$RPM"/{SOURCES,SPECS,BUILD,RPMS,SRPMS}
  install_tree "$RPM/SOURCES/root"
  sed "s/@VERSION@/$VERSION/" packaging/rpm/cherrycola.spec > "$RPM/SPECS/cherrycola.spec"
  rpmbuild --quiet -bb --define "_topdir $RPM" "$RPM/SPECS/cherrycola.spec"
  cp "$RPM"/RPMS/noarch/*.rpm "$DIST/"
else
  echo "rpmbuild not found: skipping the .rpm" >&2
fi
ls -1 "$DIST"
