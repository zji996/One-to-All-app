#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/clean_local_artifacts.sh [--venv]

Removes common local/generated artifacts inside this repo:
  - Python caches: __pycache__/ , *.pyc
  - Pytest cache: .pytest_cache/
  - Ruff / mypy caches: .ruff_cache/ , .mypy_cache/
  - Coverage artifacts: .coverage , coverage.xml , htmlcov/

Options:
  --venv    Also remove nested .venv/ directories (e.g. apps/*/.venv).
EOF
}

remove_paths() {
  local -a paths=("$@")
  local path
  for path in "${paths[@]}"; do
    if [[ -e "$path" ]]; then
      rm -rf -- "$path"
    fi
  done
}

main() {
  local remove_venv="0"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --venv) remove_venv="1" ;;
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

  remove_paths \
    ".pytest_cache" \
    ".ruff_cache" \
    ".mypy_cache" \
    ".hypothesis" \
    ".coverage" \
    "coverage.xml" \
    "htmlcov" \
    ".ipynb_checkpoints"

  # Find & remove Python caches across the repo.
  while IFS= read -r -d '' dir; do
    rm -rf -- "$dir"
  done < <(find . -type d -name '__pycache__' -print0)

  while IFS= read -r -d '' file; do
    rm -f -- "$file"
  done < <(find . -type f -name '*.py[cod]' -print0)

  if [[ "$remove_venv" == "1" ]]; then
    while IFS= read -r -d '' dir; do
      rm -rf -- "$dir"
    done < <(find . -type d -name '.venv' -print0)
  fi

  echo "Done."
}

main "$@"
