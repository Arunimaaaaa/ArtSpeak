#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h}/.."
set -a
source <(sed -E '/^[[:space:]]*#/d; /^[[:space:]]*$/d; s/^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$/\1="\2"/' "$repo_root/.env.inference")
set +a

cd "$repo_root"
if [[ -x "$repo_root/.venv-ml/bin/uvicorn" ]]; then
	uvicorn_bin="$repo_root/.venv-ml/bin/uvicorn"
else
	uvicorn_bin="$(command -v uvicorn)"
fi
exec "$uvicorn_bin" services.inference_api:app --host 127.0.0.1 --port 8000 "$@"