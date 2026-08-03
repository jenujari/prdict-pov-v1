#!/usr/bin/env sh
# Regenerate container/requirements.txt from container/requirements.in.
#
# Run this on the host after editing requirements.in, then rebuild the image.
#
# --python-platform is what makes this work at all: the host is musl and cannot
# install torch or xgboost, but cross-resolving for manylinux produces a valid
# lock for the container anyway. See kb/runtime.md.
#
# The pytorch CPU index pins torch to a +cpu build; without it the resolver
# drags in the whole CUDA stack, which is ~2 GB of nothing on a GPU-less box.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"

uv pip compile "$HERE/requirements.in" \
    --python-platform x86_64-manylinux_2_28 \
    --python-version 3.12 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    -o "$HERE/requirements.txt"

echo
echo "locked — rebuild with ./container/build.sh"
