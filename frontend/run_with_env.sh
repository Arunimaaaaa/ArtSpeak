#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h}/.."

cd "$repo_root/frontend"
exec flutter run \
  --dart-define-from-file="$repo_root/.env.frontend" \
  "$@"
