#!/usr/bin/env bash
set -euo pipefail

echo "[postStart] Fixing host-leaked gh credential helper paths (if any)..."

fix_key() {
  local key="$1"
  local existing=""
  existing="$(git config --global --get-all "$key" 2>/dev/null || true)"
  if [[ -z "$existing" ]]; then
    return 0
  fi

  local changed="0"
  if echo "$existing" | grep -q "/opt/homebrew/bin/gh auth git-credential"; then
    git config --global --unset-all "$key" "^!/opt/homebrew/bin/gh auth git-credential$" || true
    git config --global --unset-all "$key" "^/opt/homebrew/bin/gh auth git-credential$" || true
    changed="1"
  fi
  if echo "$existing" | grep -q "/usr/local/bin/gh auth git-credential"; then
    git config --global --unset-all "$key" "^!/usr/local/bin/gh auth git-credential$" || true
    git config --global --unset-all "$key" "^/usr/local/bin/gh auth git-credential$" || true
    changed="1"
  fi

  if [[ "$changed" == "1" ]]; then
    local now=""
    now="$(git config --global --get-all "$key" 2>/dev/null || true)"
    if ! echo "$now" | grep -q "^!gh auth git-credential$"; then
      git config --global --add "$key" '!gh auth git-credential'
    fi
    echo "[postStart] Updated $key to use '!gh auth git-credential'"
  fi
}

fix_key "credential.helper"
fix_key "credential.https://github.com.helper"

echo "[postStart] Done."
