#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/clean_one_to_all_animation_runtime.sh [--all]

Cleans One-to-All-Animation runtime directory under DATA_DIR (default: data/one_to_all_animation_runtime).

Default removes only model-related entries that should NOT live in runtime:
  - checkpoints (dir or symlink)
  - pretrained_models (dir or symlink)

Options:
  --all   Also remove runtime caches/workdirs (onnx_cache, video-generation).
EOF
}

main() {
  local remove_all="0"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --all) remove_all="1" ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
    esac
    shift
  done

  local repo_root
  repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
  if [[ -z "$repo_root" ]]; then
    echo "Not inside a git repo." >&2
    exit 1
  fi
  cd "$repo_root"

  local runtime_dir="${ONE_TO_ALL_RUNTIME_DIR:-${ONE_TO_ALL_ANIMATION_RUNTIME_DIR:-data/one_to_all_animation_runtime}}"
  if [[ ! -e "$runtime_dir" ]]; then
    echo "[skip] runtime not found: $runtime_dir"
    exit 0
  fi

  rm -rf -- "$runtime_dir/checkpoints" "$runtime_dir/pretrained_models"
  if [[ "$remove_all" == "1" ]]; then
    rm -rf -- "$runtime_dir/onnx_cache" "$runtime_dir/video-generation"
  fi
  echo "Done: cleaned $runtime_dir"
}

main "$@"

