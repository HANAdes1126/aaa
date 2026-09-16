#!/usr/bin/env bash
#
# Installs the built Meetly.app into /Applications, SIGNED with a stable
# self-signed code-signing identity so macOS Screen Recording (TCC) permission
# persists across rebuilds.
#
# WHY THIS MATTERS
#   A `--no-sign` build only gets a linker ad-hoc signature whose designated
#   requirement is the binary's cdhash (changes on every rebuild). TCC keys the
#   Screen Recording grant on that requirement, so a rebuild silently invalidates
#   the grant — CoreGraphics then returns a wallpaper-only frame and
#   CGPreflightScreenCaptureAccess() stays false forever. Signing with a stable
#   certificate anchors the requirement to `identifier + certificate leaf`, which
#   survives rebuilds.
#
# Usage: bash scripts/install-macos.sh
#
# First run creates a self-signed "Meetly Development" certificate in the login
# keychain (idempotent). After that, the same cert is reused every build.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

APP_SRC="$PROJECT_DIR/src-tauri/target/release/bundle/macos/Meetly.app"
APP_DST="/Applications/Meetly.app"

CERT_NAME="Meetly Development"
BUNDLE_ID="com.maidang.meetly"

info() { printf '\033[36m%s\033[0m\n' "→ $1"; }
ok() { printf '\033[32m%s\033[0m\n' "✓ $1"; }
fail() {
  printf '\033[31m%s\033[0m\n' "✗ $1" >&2
  exit 1
}

[ -d "$APP_SRC" ] || fail "Built app not found at $APP_SRC
  Build it first:
    CI=true ./node_modules/.bin/tauri build --bundles app --no-sign -c '{\"build\":{\"beforeBuildCommand\":\"npm run build\"}}'"

# --- Ensure the code-signing identity exists (idempotent) ---------------------
if ! security find-identity -v -p codesigning 2>/dev/null | grep -q "$CERT_NAME"; then
  info "Creating self-signed code-signing certificate \"$CERT_NAME\""
  TMP_DIR="$(mktemp -d)"
  openssl req -x509 -newkey rsa:2048 \
    -keyout "$TMP_DIR/key.pem" -out "$TMP_DIR/cert.pem" \
    -days 3650 -nodes -subj "/CN=$CERT_NAME" \
    -addext "extendedKeyUsage=codeSigning" \
    -addext "keyUsage=digitalSignature" >/dev/null 2>&1
  # macOS `security` cannot import modern PKCS12; use -legacy (3DES/SHA1).
  openssl pkcs12 -export -legacy -out "$TMP_DIR/cert.p12" \
    -inkey "$TMP_DIR/key.pem" -in "$TMP_DIR/cert.pem" \
    -passout pass:meetly -name "$CERT_NAME" >/dev/null 2>&1
  security import "$TMP_DIR/cert.p12" \
    -k ~/Library/Keychains/login.keychain-db -P meetly -A >/dev/null 2>&1
  security add-trusted-cert -d -r trustRoot -p codeSign \
    -k ~/Library/Keychains/login.keychain-db "$TMP_DIR/cert.pem" >/dev/null 2>&1
  rm -rf "$TMP_DIR"
  ok "Certificate created"
else
  ok "Code-signing identity \"$CERT_NAME\" present"
fi

# --- Sign the bundle with the stable identity ---------------------------------
info "Signing $APP_SRC with identifier $BUNDLE_ID"
codesign --force --deep --sign "$CERT_NAME" --identifier "$BUNDLE_ID" "$APP_SRC"
codesign --verify --deep --strict "$APP_SRC" >/dev/null 2>&1 \
  || fail "Signature verification failed"
ok "Signed (designated requirement anchors certificate, not cdhash)"

# --- Install ------------------------------------------------------------------
# A running instance holds a lock on the bundle and would survive the copy as a
# stale binary.
if pgrep -f "Meetly.app/Contents/MacOS" >/dev/null 2>&1; then
  info "Meetly is running — quitting it first"
  osascript -e 'quit app "Meetly"' 2>/dev/null || pkill -f "Meetly.app/Contents/MacOS" || true
  sleep 2
fi

info "Installing to /Applications"
if [ -d "$APP_DST" ]; then
  BACKUP_DST="/tmp/Meetly-old-$(date +%Y%m%d-%H%M%S).app"
  mv "$APP_DST" "$BACKUP_DST"
  ok "Previous app backed up: $BACKUP_DST"
fi
cp -R "$APP_SRC" /Applications/

# Unsigned/ad-hoc builds get a quarantine flag from the copy, which makes macOS
# refuse to launch them ("app is damaged").
info "Clearing quarantine attribute"
xattr -dr com.apple.quarantine "$APP_DST" 2>/dev/null || true

ok "Installed: $APP_DST"
printf '\nOpen it with:\n  open -a Meetly\n\n'
printf 'Note: if Screen Recording was granted to an OLD build, re-grant it once\n'
printf 'under System Settings → Privacy & Security → Screen Recording, then restart.\n'
printf 'The grant now persists across future rebuilds.\n'
