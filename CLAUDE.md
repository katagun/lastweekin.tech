# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

## Project

A Python pipeline that builds a weekly "Top-7" tech digest and publishes it as a static site with
an archive and an Atom feed. It runs unattended in GitHub Actions every Monday 08:00 UTC
(`.github/workflows/main.yml`), which commits the regenerated `data/`, `index.html`, `archive/`,
`feed.xml`, `sitemap.xml`, `robots.txt` and static assets back to `main`. Most commits in the
history are that bot commit.

## Commands

```bash
uv sync                                # installs the project and the dev group
uv run lastweekintech run              # full weekly run → data/ + site
uv run lastweekintech run --dry-run    # print the edition, write nothing
uv run lastweekintech ai-daily         # full AI Daily run → data/ai/ + site/ai
uv run lastweekintech ai-daily --dry-run  # print the briefing, write nothing
uv run pytest                          # 530 tests, no network
uv run pytest tests/test_curation.py -k score -q
uv run python -m evals.run             # score the summary-quality golden set
uv run python tools/check_models.py    # are the configured models still offered?
uv run ruff format . && uv run ruff check --fix .
uv run mypy src
uv run pre-commit run --all-files
```

A real run hits ~14 RSS feeds plus the Hacker News Algolia API, downloads article bodies for the
candidate pool, and calls a model per selected story — expect several minutes and set
`OPENROUTER_API_KEY` (see `.env.example`) first.

Tooling rules (from [AGENTS.md](AGENTS.md)): uv only — never Poetry; Ruff only — never
Black/Flake8. New deps go in `pyproject.toml` via `uv add`, never by hand-editing `uv.lock`.

## Architecture

`main.py` is a Typer CLI. `pipeline.build_digest()` runs the curation stages; `main` then gates,
builds the edition, saves it, writes run metrics and renders the site. Domain types are plain
dataclasses in `domain.py` (`Article`, `Story`) — no ORM, no database.

```text
fetch_articles + hn.fetch_hn_articles
  → dedupe_articles → cluster_articles → score_stories
  → discovery.fetch_consensus + apply_consensus_boost → drop_recently_published
  → [candidate pool] → extract_content → categorize_stories
  → editor.select → select_edition (or select_top_stories as fallback)
  → summarize_stories
  → assert_publishable → build_edition → save_edition → generate_site (+ syndication)
```

`build_digest` returns a `Digest` (stories plus the editor's intro), not a bare
list.

Every network boundary is an injectable parameter (`parse`, `download`, `hn_fetch`, `complete`),
which is how the suite runs without network access. Default implementations live next to their
callers.

- **hn.py** — Hacker News points come from the Algolia `search_by_date` API and are merged onto
  matching articles from other outlets by normalized URL. This is the pipeline's only non-proxy
  importance signal; the previous code scraped points from hnrss titles, which never carried them.
- **cluster_articles** — deterministic ordering, then an IDF-weighted keyword overlap using the
  week's headlines as the corpus, with `token_set_ratio` demoted to a sanity floor. Rare shared
  words are evidence that two outlets covered one event; common ones are not. Bare numbers are
  excluded as keywords (six shopping posts once merged on the year "2026").
- **score_stories** — `hn` (capped, normalized), `src` (_additional_ sources, so a lone story earns
  nothing for breadth) and `rec` (clamped to 0..1), each scaled by its configured weight. Breadth
  fires rarely — roughly 1% of stories — but is decisive when it does.
- **extract_content / extract_article_text** — bodies come from `trafilatura`, fetched with
  `requests` under a browser User-Agent. The page is extracted twice: once as-is, once with "more
  stories" containers pruned, keeping the pruned text unless it falls below `_MIN_PRUNED_BODY`.
  Without that, The Register returned its "MOST POPULAR" list as the body of every article. This
  replaced `newspaper3k`, unmaintained since 2020, which returned Slashdot's footer aphorism (82
  characters) instead of the story.
- **categorize_stories** — word-bounded matching (`text.py`) over title and body across seven
  categories in fixed precedence: AI, Security, Policy, Open Source, Hardware, Business, General
  Tech. Plain substring matching previously labelled "Britain", "train", "Hearing Aids" and "email"
  as AI.
- **discovery.py** — the consensus check: asks a Perplexity search model (`sonar-pro`) for the
  week's most important tech stories across the open web, boosts pool stories it corroborates
  (`weights.consensus`) and records unmatched headlines in the run metrics — a recurring miss is a
  missing feed. Corroborative, never generative; disabled automatically when `PERPLEXITY_API_KEY`
  is unset.
- **editor.py** — editorial selection: one model call a week (default `anthropic/claude-sonnet-5`,
  with fallbacks) reads the candidate pool with a rubric and picks the edition in print order,
  with a one-line `why` per story and a 2-3 sentence `intro` for the week. The mechanical guards
  stay authoritative: its picks pass the AI floor, the source cap and the body preference in
  `select_edition`, and any failure falls back to `select_top_stories`. `max_tokens` is generous
  (8000) because reasoning models think out of the same budget — 2000 truncated the first live
  verdict.
- **select_top_stories** — `min_ai` is a floor, not a quota: promotion happens only when the ranking
  falls short, and displaces the weakest general stories. Two further preferences both yield rather
  than shorten an edition: stories with extracted bodies come first (a bodyless story cannot be
  summarized, which is how the 2026-08-24 run failed the gate), and `digest.max_per_source` caps
  slots per displayed outlet — WIRED once supplied 40% of everything published, and the first
  HN-scored edition was seven of seven Hacker News.
- **Aggregator feeds** (Techmeme, `aggregator: true` in config.yaml) corroborate but never lead:
  they count toward breadth, yet `_representative_article` links the original outlet and
  `summarize_stories` prefers original bodies — summarizing Techmeme's aggregation page produced
  a live refusal.
- **drop_recently_published** — excludes stories from the last `digest.repeat_lookback_weeks`
  editions by URL and near-identical title, with a floor that keeps the edition full.
- **summarize_stories** — tries every article in the cluster, original outlets before aggregators,
  richest body first; leaves `summary` unset when nothing works so the gate can see it.
  `Summarizer.last_model` records which model in the fallback chain actually answered, which is
  what the metrics report. The summarizer strips leading markdown furniture and rejects refusals
  ("I cannot summarize…") so the fallback chain retries instead of shipping them.
- **quality.py** — deterministic, network-free checks on a summary (truncation, subject coverage,
  number and entity grounding, contract, length). Pointed at the 308 archived summaries it
  reproduces the failure rates measured by hand: 48% empty, 23% truncated.
- **validation.py** — the publish gate. Every other stage degrades gracefully, so this is the one
  place that refuses to ship. It rejects missing summaries outright and consults `quality.py` for
  present-but-broken ones; `BLOCKING_CHECKS` excludes `LENGTH` because a terse summary is off-style,
  not wrong.
- **metrics.py** — a `RunMetrics` record per run, written to `data/runs/<week>.json` even when the
  gate rejects the edition. It carries `ai_before_promotion` vs `ai_promoted`, which is the data
  needed to decide whether the AI floor still earns its place.
- **syndication.py** — Atom feed (one entry per edition, `tag:` URIs for stable ids), sitemap and
  robots.txt. Stdlib only, escaping applied twice on purpose for `type="html"` content.
- **generate_site** — renders `templates/edition.html.jinja` for the latest edition and every
  archived one, then copies every file in `static/`. Paths resolve from `PACKAGE_DIR`, so the
  pipeline works from any working directory. Autoescape is unconditional: `select_autoescape()`
  keys off the file extension and left `.html.jinja` templates unescaped.
- **briefer.py** — AI Daily's selection-and-analysis model call. Unlike the weekly pipeline's
  editor+summarizer split, one call does both jobs: there is no mechanical fallback that can
  substitute for AI-generated analysis, so a verdict is either fully valid — right pick-count
  range, every field populated, per-source cap and article-text presence both mechanically
  enforced — or the whole call is treated as failed and the next model in the chain is tried.
- **ai_daily.py** — AI Daily's orchestration and storage. Reuses the weekly pipeline's
  fetch/dedupe/cluster/score/extract stages unmodified via a derived `Config` (a 24-hour window,
  daily-tuned HN thresholds). No mechanical category filter narrows the candidate pool — the
  fixed-precedence classifier would tag a chip-export-control or policy story as Hardware or
  Policy rather than AI — so relevance is judged entirely by the briefing model against a broad
  pool. It renders its own site section, feed and sitemap at `/ai/`.
- AI Daily has no publish gate: a `Briefer` failure across its whole fallback chain fails the
  day's run closed (nothing published, the previous day's page stays live), a deliberate
  divergence from the weekly pipeline's everything-degrades-gracefully-until-the-gate philosophy
  — there is no partial verdict for a gate like `validation.py` to arbitrate.

Everything under `public/` is **generated output** — edit `src/lastweekintech/templates/` and
`src/lastweekintech/static/` instead. The site is generated into its own directory rather than the
repository root because GitHub Pages uploads that tree verbatim as the deploy artifact, so anything
outside it is never served. `tools/` holds one-off scripts that are deliberately _not_ in
`static/`, because everything in `static/` gets published.

## Testing

`tests/conftest.py` holds `make_article`, `make_story` and the `config` fixture; `NOW` is a fixed
timestamp so time-dependent assertions are exact. Tests import helpers from sibling modules
directly (`pythonpath = ["tests"]`).

When changing curation behaviour, note that test headlines must be genuinely distinct or the
clusterer will merge them and the assertion will be about clustering rather than what you meant to
test — `test_build_digest.SUBJECTS` exists for this.

`evals/` is separate from the unit tests: a golden set of summaries with expected verdicts, run
with `uv run python -m evals.run`. Its `--judge` mode calls a model and is never exercised by the
suite.

## Data

`data/archive/*.json` holds every published edition (backfilled from git history; categories there
were recomputed with the fixed classifier). `data/latest.json` is a copy of the newest one.
`data/runs/*.json` holds per-run metrics. `data/ai/archive/*.json` holds every published AI Daily
edition the same way, and `data/ai/latest.json` is a copy of the newest one.

## Operations

Model IDs churn — three of the four models this project shipped with had already been retired by
the provider. `tools/check_models.py` verifies the configured chain against the live catalogue and
runs on a schedule the day before each digest. A failed digest opens (or comments on) a GitHub
issue rather than sending an email nobody reads.

When the scheduled run cannot publish, `scripts/publish_fallback.sh` reproduces it locally (checks
included; `--push` commits to main), and the `publish-digest` project skill walks that procedure —
diagnose first, run without publishing, review, push only with an explicit go-ahead.

## Spec

`docs/lwit-spec-architecture.md` is the original product spec. It predates the implementation in
places: it specifies LiteLLM (the code uses the OpenAI SDK against OpenRouter), local summarization
models, deployment to Vercel (the site is on GitHub Pages), and a SQLite store that was never used
and has been removed. Treat the goals as current and the tech-stack sections as historical — its
Vercel references describe a plan that was never carried out, and are left in place as the record
of the original design.

## Outside review

`docs/review-2026-08-17.md` records an external review from 2026-08-17: the domain
did not resolve, the publish gate was not enforcing `max_missing_summaries`, and the
empty-summary rate had improved but plateaued. The first two are fixed (the site is
live on GitHub Pages; the gate enforces the limit) and the third is still worth
watching — the file carries a status banner saying so. It is a dated record, not a
current to-do list.
