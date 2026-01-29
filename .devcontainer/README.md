# Dev Container notes

This repo is designed to run the autonomous swarm (`python scripts/swarm.py ...`) inside a devcontainer/VM/Codespaces sandbox.

## Yarn apt key (build fix)

The base image (`mcr.microsoft.com/devcontainers/python:1-3.11-bookworm`) includes Yarn’s apt repo.
In late January 2026 the repo signing key rotated, which can break `apt-get update` during devcontainer feature installation
with `NO_PUBKEY 62D54FD4003F6525`.

This devcontainer uses `.devcontainer/Dockerfile` to refresh the Yarn apt keyring before features run.

## Why `--security-opt=seccomp=unconfined`

Codex CLI’s Linux sandbox relies on Landlock + seccomp. In some Docker/devcontainer setups, the container’s default seccomp profile blocks the syscalls Codex needs, which can surface as errors like:

`error running landlock: Sandbox(LandlockRestrict)`

This devcontainer sets `runArgs: ["--security-opt=seccomp=unconfined"]` so Codex can apply its own sandbox inside the container.

If your host still can’t support Landlock/seccomp in Docker, a fallback is to run the swarm with:

`--codex-sandbox danger-full-access`

and rely on the devcontainer boundary for isolation.

## GitHub CLI (`gh`) credential helper

VS Code may copy your host `~/.gitconfig` into the container. If that config contains a macOS Homebrew absolute path for `gh` (e.g. `/opt/homebrew/bin/gh`), git operations can spam `gh: not found` inside the Linux container.

On container start, `.devcontainer/postStart.sh` replaces any host-leaked Homebrew `gh auth git-credential` helpers with:

`!gh auth git-credential`

so the `gh` found on the container `PATH` is used.
