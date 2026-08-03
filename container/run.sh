#!/usr/bin/env sh
# Run a command inside the ML container with the repo bind-mounted at /work.
#
# The host is musl and cannot install torch or xgboost (see kb/runtime.md), so
# every model-related step goes through here.
#
#   ./container/run.sh python container/verify.py
#   ./container/run.sh python scripts/train_xgboost.py
#   ./container/run.sh                     # interactive shell
#
# Build or rebuild the image with ./container/build.sh
set -eu

IMAGE="${PRDICT_IMAGE:-localhost/prdict-ml:latest}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

if ! podman image exists "$IMAGE"; then
    echo "image $IMAGE not found — run ./container/build.sh first" >&2
    exit 1
fi

# Rootless podman maps the container's root to the invoking host user, so files
# written to /work come back owned by that user with no --userns juggling.
if [ "$#" -eq 0 ]; then
    set -- /bin/bash
fi

# Only allocate a TTY when there is one — otherwise podman warns on every
# scripted or CI invocation.
TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-t"

exec podman run --rm -i $TTY_FLAG \
    -v "$REPO":/work:rw \
    -w /work \
    "$IMAGE" "$@"
