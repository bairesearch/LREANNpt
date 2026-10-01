#!/usr/bin/env bash
set -euo pipefail

# Forward arguments to the launcher; defaults to this script's folder from any working directory.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$script_dir/reproduce.py" "$@"
