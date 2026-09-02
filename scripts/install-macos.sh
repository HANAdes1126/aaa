#!/usr/bin/env bash
#
# Installs the built Meetly.app into /Applications so it can be opened by
# double-click, with no terminal and no dev server.
#
# Usage: bash scripts/install-macos.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

APP_SRC="$PROJECT_DIR/src-tauri/target/release/bundle/macos/Meetly.app"
APP_DST="/Applications/Meetly.app"

info() { printf '\033[36m%s\033[0m\n' "→ $1"; }
ok() { printf '\033[32m%s\033[0m\n' "✓ $1"; }
fail() {
  printf '\033[31m%s\033[0m\n' "✗ $1" >&2
  exit 1
}

[ -d "$APP_SRC" ] || fail "Built app not found at $APP_SRC
  Build it first:
    CI=true ./node_modules/.bin/tauri build --bundles app --no-sign -c '{\"build\":{\"beforeBuildCommand\":\"npm run build\"}}'"

# A running instance holds a lock on the bundle and would survive the copy as a
# stale binary.
if pgrep -f "Meetly.app/Contents/MacOS" >/dev/null 2>&1; then
  info "Meetly is running — quitting it first"
  osascript -e 'quit app "Meetly"' 2>/dev/null || pkill -f "Meetly.app/Contents/MacOS" || true
  sleep 2
fi

info "Installing to /Applications"
rm -rf "$APP_DST"
cp -R "$APP_SRC" /Applications/

# Unsigned builds get a quarantine flag from the copy, which makes macOS refuse
# to launch them ("app is damaged").
info "Clearing quarantine attribute"
xattr -dr com.apple.quarantine "$APP_DST" 2>/dev/null || true

ok "Installed: $APP_DST"
printf '\nOpen it with:\n  open -a Meetly\n\n'
printf 'Tip: drag /Applications/Meetly.app onto the Dock to keep it pinned.\n'
printf 'Ghost mode and its sliders live in Settings → 外观与隐身.\n'
