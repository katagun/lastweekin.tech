#!/usr/bin/env bash
# Manual fallback publisher, for when the scheduled GitHub Actions run cannot.
#
# Reproduces what .github/workflows/main.yml does on a Monday: run the
# pipeline, then commit the regenerated data/ and public/ trees to main and
# push. Without --push it stops after the pipeline run so the edition can be
# reviewed first; nothing is committed or pushed.
#
# Usage:
#   scripts/publish_fallback.sh           # run the pipeline, write data/ and public/, do not push
#   scripts/publish_fallback.sh --push    # commit the regenerated output to main and push it
#   scripts/publish_fallback.sh --dry-run # print the edition, write nothing (forwarded to the CLI)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

push=false
cli_args=()
for arg in "$@"; do
  case "$arg" in
    --push) push=true ;;
    *) cli_args+=("$arg") ;;
  esac
done

# The published site is built from main; publishing from anywhere else would
# fork the archive.
branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != "main" ]; then
  echo "Refusing to run: on branch '$branch', not main." >&2
  exit 1
fi

git fetch origin main --quiet
if [ -n "$(git rev-list HEAD..origin/main)" ]; then
  echo "Refusing to run: main is behind origin/main. Pull first." >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing to run: the working tree is not clean." >&2
  exit 1
fi

week=$(date -u +%Y-%m-%d)
if git cat-file -e "origin/main:data/archive/$week.json" 2>/dev/null; then
  echo "An edition for $week is already published on origin/main; nothing to do."
  exit 0
fi

# Secrets: the environment wins; .env fills the gaps for a local run.
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is not set (and no .env provides it)." >&2
  exit 1
fi

uv sync --quiet
uv run lastweekintech run "${cli_args[@]}"

if [ "$push" != true ]; then
  echo
  echo "Pipeline finished. Review data/latest.json and public/index.html,"
  echo "then re-run with --push to commit and publish."
  exit 0
fi

git add -- data public
if git diff --staged --quiet; then
  echo "No changes to commit."
  exit 0
fi
git commit -m "docs: Update weekly digest for $week (manual fallback)"
git push origin main
echo "Published the $week edition to main."
