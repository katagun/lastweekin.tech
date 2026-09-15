# LastWeekIn.Tech

## Overview

LastWeekIn.Tech is a Python pipeline that publishes a weekly "Top-7" tech
digest as a static site. It aggregates RSS feeds and Hacker News, deduplicates
coverage of the same event into stories, ranks them, summarizes each with an
LLM, and writes the result as JSON plus a generated `index.html`. It runs
unattended in GitHub Actions every Monday at 08:00 UTC.

## Features

- **Multi-source aggregation**: RSS and Atom feeds plus Hacker News, over a
  configurable time window.
- **Real importance signal**: Hacker News point totals come from the Algolia
  search API and are matched onto the same story as covered by other outlets.
- **Story clustering**: fuzzy title matching groups multiple outlets' coverage
  of one event, with exact-URL deduplication first.
- **Transparent scoring**: breadth of coverage, Hacker News traction and
  recency, each normalized to 0..1 before its configured weight applies.
- **AI coverage floor**: at least half the digest is AI-related, promoted from
  the ranking only when the week falls short — never capped.
- **LLM summaries**: any OpenAI-compatible endpoint, with an ordered model
  fallback chain and sentence-boundary trimming of truncated output.
- **Publish gate**: a digest that is short, duplicated or missing summaries
  fails the run instead of replacing last week's edition.
- **Archive**: every edition is kept as JSON and gets its own page.
- **Subscribable**: an Atom feed, a sitemap and social cards — no email required.
- **Measurable**: deterministic summary-quality checks with a golden set, and a
  per-run metrics record in `data/runs/`.

## Setup

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Then edit `.env`:

- `OPENROUTER_API_KEY` — required, from [OpenRouter](https://openrouter.ai/).
- `HF_TOKEN` — optional, only for the Hugging Face fallback models.

`uv sync` installs the dev tooling too, so `uv run pytest`, `uv run ruff` and
`uv run mypy` work immediately.

## Usage

```bash
uv run lastweekintech run
```

This writes the edition to `data/` and the whole static site to `public/` —
`index.html`, `archive/<week>.html`, `feed.xml`, `sitemap.xml`, `robots.txt`,
`404.html` and the static assets. The pipeline resolves its templates from the
installed package, so it can be run from any directory.

### Options

| Option             | Description                                                       |
| ------------------ | ----------------------------------------------------------------- |
| `--data-dir`, `-d` | Where `latest.json` and the archive are written (default `data`). |
| `--site-dir`, `-s` | Where the static site is written (default `public`).              |
| `--config`, `-c`   | Path to an alternative `config.yaml`.                             |
| `--week`, `-w`     | Edition date, `YYYY-MM-DD`. Defaults to today in UTC.             |
| `--dry-run`        | Print the edition as JSON and write nothing.                      |
| `--skip-gate`      | Publish even if validation fails.                                 |

`--dry-run` is the quickest way to see what a run would publish:

```bash
uv run lastweekintech run --dry-run
```

## Development

```bash
uv run pytest                     # full suite, no network
uv run pytest tests/test_curation.py::TestScoreStories -q   # one class
uv run python -m evals.run        # score the summary-quality golden set
uv run python tools/check_models.py   # are the configured models still offered?
uv run ruff format . && uv run ruff check --fix .
uv run mypy src
uv run pre-commit run --all-files
```

Regenerating the social card needs Pillow, which is deliberately not a declared
dependency:

```bash
uv run --with pillow python tools/make_og_image.py
```

CI runs lint, format check, mypy, bandit and the test suite on every push and
pull request.

## How the pipeline works

```text
RSS feeds ─┐
           ├─→ dedupe by URL ─→ cluster by title ─→ score ─→ candidate pool
HN Algolia ┘                                                      │
                                                                  ▼
     publish ←─ gate ←─ summarize ←─ select top 7 ←─ categorize ←─ extract bodies
```

Article bodies are downloaded only for the candidate pool, after ranking —
fetching every article of the week to decide seven slots is the slowest thing
the pipeline could do. Downloads run concurrently across hosts and are spaced
per host to stay polite. Body text is extracted with
[trafilatura](https://trafilatura.readthedocs.io/); a page that yields nothing
extractable (a PDF, a plain text file, a JavaScript-rendered shell) keeps its
place in the story with no content rather than being dropped.

## Configuration

`src/lastweekintech/config.yaml` drives everything. `${VAR}` values are read
from the environment at load time; an unset variable becomes `null`, which
disables the feature rather than passing a placeholder as a credential.

| Section       | Purpose                                                                                                                           |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `feeds`       | RSS/Atom sources. Hacker News is not listed here — see `hn`.                                                                      |
| `hn`          | Algolia lookup: `enabled`, `min_points`, `points_cap`, `limit`.                                                                   |
| `window_days` | How far back to look.                                                                                                             |
| `weights`     | Relative importance of `hn`, `src` (breadth) and `rec` (recency).                                                                 |
| `digest`      | `story_count`, `min_ai_stories`, `max_missing_summaries`, `max_low_quality_summaries`, `candidate_pool`, `repeat_lookback_weeks`. |
| `site_url`    | Absolute origin, used for the feed, sitemap, canonical links and cards.                                                           |
| `summarizer`  | Model chain, `max_tokens`, `max_input_chars`, `temperature`.                                                                      |

Unknown keys are rejected at load time, so a typo fails the run instead of
being silently ignored.

## Deployment

The site is served from `public/` by GitHub Pages at
<https://lastweekin.tech>, and that directory is the whole of what is
published. It is deliberately separate from the repository root: the Pages
artifact is uploaded from `public/` alone, so `src/`, `tests/`, `data/` and
`evals/` are never served from the site.

`.github/workflows/deploy-pages.yml` is the deploy. It runs on any push to
`main` that touches `public/**`, and — because the weekly digest commits with
the workflow's own `GITHUB_TOKEN`, which never fires a push event — also on
completion of the **Weekly Tech Digest** workflow. Pages is configured for
`build_type: workflow`, so nothing is built at deploy time; the tree ships as
generated.

Pages serves static files with fixed headers and no per-path configuration, so
two things are worth knowing: `feed.xml` is served as `application/xml` rather
than the registered `application/atom+xml` (readers accept it, but it is not
the canonical type), and the URLs in the sitemap and feed are served verbatim
with no `cleanUrls`-style rewriting — which is what those files assume.

## Automation

`.github/workflows/main.yml` runs the digest weekly and commits the result. It
needs `OPENROUTER_API_KEY` as a repository secret (Settings → Secrets and
variables → Actions), and optionally `HF_TOKEN`. If the publish gate rejects
the digest, the job fails and the previous edition stays live; re-run the
workflow manually with **Publish even if the digest fails validation** to
override.
