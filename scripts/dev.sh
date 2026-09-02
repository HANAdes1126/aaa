#!/usr/bin/env bash
#
# Starts Meetly in Tauri dev mode.
#
# The documented flow is `pnpm install && pnpm tauri dev`, but pnpm is broken on
# some sandboxed setups (sqlite "disk I/O error"), and tauri.conf.json hardcodes
# `pnpm dev` / `pnpm build` in beforeDevCommand. Rather than editing project
# config, this script overrides those values through the CLI `--config` flag,
# which merges on top of tauri.conf.json.
#
# Usage: bash scripts/dev.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Managed Node lives outside the default PATH.
NODE_BIN="/Users/zhanweixiang/.workbuddy/binaries/node/versions/22.22.2-2/bin"
[ -d "$NODE_BIN" ] && export PATH="$NODE_BIN:$PATH"
[ -d "$HOME/.cargo/bin" ] && export PATH="$HOME/.cargo/bin:$PATH"

fail() {
  printf '\033[31m%s\033[0m\n' "✗ $1" >&2
  exit 1
}

info() { printf '\033[36m%s\033[0m\n' "→ $1"; }

# --- Preflight ---------------------------------------------------------------

command -v cargo >/dev/null 2>&1 || fail "Rust toolchain not found. Install with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
command -v npm >/dev/null 2>&1 || fail "npm not found."
[ -d node_modules ] || fail "Dependencies not installed. Run: npm install --no-package-lock"

# cidre's build script shells out to xcodebuild, so the full Xcode app is
# required — Command Line Tools alone will fail at the cidre compile step.
if ! xcode-select -p 2>/dev/null | grep -q "Xcode.app"; then
  fail "Full Xcode is required (not just Command Line Tools).
  Install with: xcodes install --latest
  Then select:  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
fi

if ! xcodebuild -license check >/dev/null 2>&1; then
  fail "Xcode license not accepted. Run: sudo xcodebuild -license accept"
fi

# `tauri dev` starts its own Vite server on 1420; a stale one blocks startup.
if lsof -ti:1420 >/dev/null 2>&1; then
  info "Port 1420 in use — stopping the stale dev server"
  lsof -ti:1420 | xargs kill -9 2>/dev/null || true
  sleep 1
fi

# --- Run --------------------------------------------------------------------

info "Starting Tauri dev (beforeDevCommand overridden to 'npm run dev')"
exec ./node_modules/.bin/tauri dev -c '{"build":{"beforeDevCommand":"npm run dev"}}'
