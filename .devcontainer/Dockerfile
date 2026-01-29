FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

# The base devcontainers python image includes Yarn's apt repo.
# As of 2026-01-29, the signing key used by that repo rotated and older base images
# fail `apt-get update` with:
#   NO_PUBKEY 62D54FD4003F6525
#
# Update the keyring so devcontainer "features" (which run apt-get) can install.
RUN set -eu; \
  export GNUPGHOME="$(mktemp -d)"; \
  curl -fsSL https://dl.yarnpkg.com/debian/pubkey.gpg -o /tmp/yarn-pubkey.gpg; \
  gpg --dearmor --batch --yes -o /usr/share/keyrings/yarn-archive-keyring.gpg /tmp/yarn-pubkey.gpg; \
  rm -f /tmp/yarn-pubkey.gpg; \
  rm -rf "$GNUPGHOME"
