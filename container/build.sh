#!/usr/bin/env sh
# Build the ML container image.
#
# Build context is container/ only, so the 28 MB nft50.csv and the 48 MB PDFs in
# kb/ never enter the build.
#
# --network=host: this machine resolves DNS through tailscale (100.100.100.100),
# which podman's build network namespace cannot reach — without it the base
# image pull fails with "lookup ghcr.io ... no such host".
set -eu

IMAGE="${PRDICT_IMAGE:-localhost/prdict-ml:latest}"
HERE="$(cd "$(dirname "$0")" && pwd)"

podman build --network=host -t "$IMAGE" -f "$HERE/Containerfile" "$HERE"
echo
echo "built $IMAGE — verify with: ./container/run.sh python container/verify.py"
