#!/bin/sh
# Compatibility entry point; v5.2.1 uses one complete regression collection.
set -eu
cd "$(dirname "$0")/.."
exec sh scripts/run_regression.sh "${1:-verification-local}"
