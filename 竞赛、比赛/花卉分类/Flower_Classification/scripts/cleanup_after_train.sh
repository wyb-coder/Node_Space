#!/usr/bin/env bash
# Cleanup script for trained artifacts
# - archives results/model and results/results into a timestamped tar.gz (if they exist)
# - removes results/model, results/results directories and model/temp_best_model.pth
# Usage:
#   bash scripts/cleanup_after_train.sh    # interactive confirmation
#   bash scripts/cleanup_after_train.sh --yes   # non-interactive

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

BACKUP_NAME="results_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
DRY_RUN=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--yes] [--dry-run]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 2
      ;;
  esac
done

TO_REMOVE=("results/model" "results/results" "model/temp_best_model.pth")
EXISTING=()
for p in "${TO_REMOVE[@]}"; do
  if [ -e "$p" ]; then
    EXISTING+=("$p")
  fi
done

if [ ${#EXISTING[@]} -eq 0 ]; then
  echo "No target files or directories found: nothing to do."
  exit 0
fi

echo "The following will be archived (if directories) and removed:" 
for e in "${EXISTING[@]}"; do
  echo "  - $e"
done

if [ $DRY_RUN -eq 1 ]; then
  echo "Dry-run mode: no changes will be made."
  exit 0
fi

if [ $FORCE -ne 1 ]; then
  read -p "Proceed with backup+remove? [y/N]: " yn
  case "$yn" in
    [Yy]*) ;;
    *) echo "Aborted by user."; exit 1 ;;
  esac
fi

# Build tar list for directories only
TAR_LIST=()
for e in "${EXISTING[@]}"; do
  if [ -d "$e" ]; then
    TAR_LIST+=("$e")
  fi
done

if [ ${#TAR_LIST[@]} -gt 0 ]; then
  echo "Creating backup archive: $BACKUP_NAME"
  tar -czf "$BACKUP_NAME" "${TAR_LIST[@]}"
  echo "Backup created: $BACKUP_NAME"
else
  echo "No directories to archive. Skipping tar backup."
fi

# Now remove targets
for e in "${EXISTING[@]}"; do
  echo "Removing $e"
  rm -rf "$e"
done

echo "Cleanup complete. If you need to restore, extract $BACKUP_NAME and copy files back." 
exit 0
