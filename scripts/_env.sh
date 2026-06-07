# shellcheck shell=bash
# Resolve a Python interpreter that has the project dependencies installed.
# `uv sync` installs into ./.venv, but a bare `python` on the PATH (system or
# conda) won't see those packages. Preference order:
#   1. $NEWSREC_PYTHON (explicit override)
#   2. ./.venv/bin/python   (the uv-managed project venv)
#   3. `uv run python`      (let uv resolve/activate the env)
#   4. python               (last resort; PATH interpreter)
# Run scripts from the repo root (the scenario scripts `cd` there first).
if [ -n "${NEWSREC_PYTHON:-}" ]; then
  PY="${NEWSREC_PYTHON}"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  PY="uv run python"
else
  PY="python"
fi
echo "[newsrec] using interpreter: ${PY}"
