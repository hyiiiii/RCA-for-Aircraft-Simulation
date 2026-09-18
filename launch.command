#!/bin/bash
set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

usable_python() {
  local py="$1"
  [ -n "$py" ] && [ -x "$py" ] && "$py" -c "import numpy, tkinter" >/dev/null 2>&1
}

PYTHON=""
for candidate in \
  "$PROJECT_DIR/.venv/bin/python" \
  "${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}" \
  "$HOME/miniconda3/envs/causal_discovery/bin/python" \
  "$HOME/miniconda3/envs/aircraft-assembly-simulation/bin/python" \
  "/opt/miniconda3/envs/causal_discovery/bin/python" \
  "/opt/miniconda3/envs/aircraft-assembly-simulation/bin/python"
do
  if usable_python "$candidate"; then
    PYTHON="$candidate"
    break
  fi
done

if [ -z "$PYTHON" ] && command -v python3 >/dev/null 2>&1 && python3 -c "import numpy, tkinter" >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
fi

if [ -z "$PYTHON" ]; then
  printf '%s\n' '未找到同时具备 numpy 和 Tkinter 的 Python。请按 README 创建环境后再打开本文件。'
  exit 1
fi

exec "$PYTHON" "$PROJECT_DIR/industrial_desktop_app.py" "$@"
