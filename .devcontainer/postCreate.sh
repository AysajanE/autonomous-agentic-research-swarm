#!/usr/bin/env bash
set -euo pipefail

echo "[postCreate] Installing base utilities (tmux, jq)..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends tmux jq

echo "[postCreate] Verifying/Installing GitHub CLI (gh)..."
if ! command -v gh >/dev/null 2>&1; then
  # Fallback: install from distro if the devcontainer feature isn't available for any reason.
  sudo apt-get install -y --no-install-recommends gh
fi

echo "[postCreate] Installing/Updating Codex CLI..."
npm i -g @openai/codex@latest

echo "[postCreate] gh: $(gh --version | head -n 1 || true)"

echo "[postCreate] Done."
