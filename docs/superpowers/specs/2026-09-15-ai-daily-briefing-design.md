# AI Daily Briefing — design

Status: approved for planning
Date: 2026-09-15

## Purpose

A second, independent pipeline alongside the weekly Top-7 digest: once a day, find the
4–5 most important AI news developments from the past 24 hours and publish a short
analytical brief for each — what happened, why it matters (including near-term
implications), and what to watch next. Scope is AI capabilities, major companies,
open-source models, regulation/policy, security/safety, infrastructure/chips, and
industry economics. Low-impact product announcements and repetitive stories are
excluded unless they materially change the picture.

It reuses as much of the existing weekly pipeline's infrastructure as the differences
in cadence, scope and content depth allow, rather than forking a parallel codebase.

## Naming and surface

- Feature name: **AI Daily**
- URL: `lastweekin.tech/ai/` (latest) and `lastweekin.tech/ai/archive/<date>.html`
- New module: `src/lastweekintech/ai_daily.py`
- New template: `src/lastweekintech/templates/ai_daily.html.jinja`
- New CLI subcommand: `uv run lastweekintech ai-daily`
- New workflow: `.github/workflows/ai-daily.yml`, daily cron at 13:00 UTC
- Discoverability: a single small link from the main site (header or footer) to `/ai/`.
  No homepage feature, no prominent promotion — "subtle" per the request that started
  this design.
- Archive and Atom feed: full parity with the weekly digest (see "Storage" and
  "Syndication reuse" below), not a latest-only page.

## Data flow

Reuses these existing functions from `pipeline.py` and `hn.py` **unmodified**:
`fetch_articles`, `hn.fetch_hn_articles`, `dedupe_articles`, `cluster_articles`,
`score_stories`, `extract_content`. Reuse is possible because each of these takes the
whole `Config` object (or values already on it) rather than reading module-level
config — so a derived `Config` (via `dataclasses.replace`) with AI-daily-specific
`window_days`, `hn`, and `weights` fields flows through them without any code changes.

`categorize_stories` (the fixed-precedence keyword classifier) is **not** used for
this pipeline. Its categories (AI / Security / Policy / Open Source / Hardware /
Business / General Tech) are a mechanical proxy tuned for weekly breadth-of-topic
selection, not an AI-relevance judge — a chip export-control story would land under
Hardware or Policy, not AI, and would be invisible to a category-AI filter. Relevance
here is judged entirely by the briefing LLM call (see "Briefing step") against a
broad, unfiltered candidate pool.

Pipeline shape:

```text
fetch_articles + hn.fetch_hn_articles   (window_days=1, ai_daily.hn settings)
  → dedupe_articles → cluster_articles → score_stories (ai_daily.weights)
  → drop_recently_published (lookback: timedelta(days=ai_daily.repeat_lookback_days),
                              keep_at_least=ai_daily.story_count)
  → [candidate pool, ai_daily.candidate_pool] → extract_content
  → Briefer.brief(...)   (selection + full analysis in one LLM call)
  → build_ai_edition → save_ai_edition → generate_ai_site (+ ai feed/sitemap)
```

**Known tradeoff**: `score_stories` still bounds the candidate pool (default 30)
*before* the briefing LLM ever sees the pool. A high-importance, low-HN-traction story
(e.g. a policy announcement with modest engagement but real significance) could in
principle be pre-filtered out before the LLM has a chance to judge it. This mirrors a
tension already present in the weekly pipeline (mechanical ranking bounds the pool the
editor LLM chooses from) and is mitigated, not eliminated, by using a generous pool
size relative to daily's naturally smaller volume. If this proves to be a real problem
in practice, a future iteration could fold the existing Perplexity consensus check
(`discovery.py`) into the daily scoring stage too — deliberately deferred for v1 to
keep scope contained.

## Config

New top-level `ai_daily` section in `config.yaml`, and a matching `AiDailySettings`
dataclass (plus a nested `AiBriefingSettings`) in `config.py`:

```yaml
ai_daily:
  enabled: true
  window_days: 1
  candidate_pool: 30
  story_count: 5
  min_story_count: 4       # floor, not a pad — never invents a 5th pick
  repeat_lookback_days: 3
  max_per_source: 2
  hn:
    min_points: 20          # 24h HN points run far lower than a week's worth
    points_cap: 300
    limit: 200
  weights:
    hn: 5
    src: 3
    rec: 1
  briefing:
    model_name: "anthropic/claude-sonnet-5"
    fallback_models: ["anthropic/claude-haiku-4.5", "google/gemini-3.7-flash"]
    max_tokens: 6000
    temperature: 0.3
    excerpt_chars: 600
```

`Config.from_yaml`'s `unknown` key check and `validate()` both need to accept and
validate this new section (mirroring how `editor`/`perplexity` were added previously):
`min_story_count <= story_count`, `candidate_pool >= story_count`, non-empty
`briefing.model_name`.

**Shared-code cleanup this motivates**: `drop_recently_published`'s
`lookback_weeks: float` parameter becomes `lookback: timedelta`. Forcing "3 days" into
a weeks-shaped parameter is an awkward unit mismatch; both call sites become clearer
(`timedelta(weeks=config.digest.repeat_lookback_weeks)` for weekly,
`timedelta(days=config.ai_daily.repeat_lookback_days)` for daily). This is the only
change to shared pipeline code this feature requires.

## Domain

A new `AiBrief` dataclass in `domain.py`, distinct from `Story`:

```python
@dataclass
class AiBrief:
    """One AI Daily item: a story plus its analysis."""

    title: str
    articles: list[Article] = field(default_factory=list)
    theme: str = ""              # one of the seven fixed themes, see below
    what_happened: str = ""
    why_it_matters: str = ""
    watch_next: str = ""
    score: float = 0.0
```

`Story` is not reused for this content because the shapes genuinely differ: `Story`
carries one summary and one optional one-line "why"; `AiBrief` carries three
substantial text sections plus a theme. Forcing one shape to cover both would leave
unused fields on one side or the other.

## Briefing step

A new `Briefer` class in `ai_daily.py`, structurally parallel to `Editor` and
`Summarizer` (same `CompleteFn`/`Completion` injection from `summarizer.py`, so it is
testable offline the same way; same fallback-across-models chain).

### Prompt

Embeds the exact selection criteria from the original request: prioritize AI
capabilities, major companies, open-source models, regulation/policy, security/safety,
infrastructure/chips, and industry economics; explicitly downrank low-impact product
announcements; avoid repetitive stories unless they materially change the picture. The
"avoid repeating" instruction is backed by a short list built from the AI Daily
archive: titles published within `repeat_lookback_days`, given to the model as context
("recently covered, skip unless this is a material update").

Mechanical dedupe (`drop_recently_published`, exact URL/near-identical title match)
still runs first, as a floor — the LLM judgment layer above it is for near-duplicates
and rehashes a title-similarity check would miss, not a replacement for it.

### Response schema

JSON-only reply, like `editor.py`:

```json
{
  "intro": "1-2 sentences on the day's overall AI news shape",
  "picks": [
    {
      "n": 7,
      "theme": "Infrastructure & Chips",
      "what_happened": "2-4 factual sentences, sourced only from the article.",
      "why_it_matters": "Analysis paragraph: significance and near-term implications.",
      "watch_next": "1-2 sentences on what to watch for next."
    }
  ]
}
```

`theme` is constrained in the prompt to a fixed enum of seven values matching the
request's own categories: `Capabilities`, `Companies`, `Open Source`,
`Policy & Regulation`, `Security & Safety`, `Infrastructure & Chips`,
`Industry & Economics`.

### Validation and failure handling

No new `quality.py`/`validation.py`-style subsystem for v1. `Briefer.brief()` applies
the same "trustworthy or bust" bar `Editor.select()` already uses: the response is
accepted only if it has between `min_story_count` and `story_count` distinct, in-range
picks, each with a non-empty `what_happened`, `why_it_matters`, and a `theme` from the
fixed enum. Anything less (wrong count, missing field, `theme` outside the enum,
unparseable JSON) is treated as a failed call and the chain moves to the next model.

This is a deliberate divergence from the weekly pipeline's philosophy. Weekly degrades
gracefully at every stage and only refuses to publish at the very end (the gate); here,
there is no mechanical fallback that can produce analysis prose, so a `Briefer` call
that exhausts its whole fallback chain (primary + all fallback models) **fails the run
for the day**: nothing is published, yesterday's page stays live, and a GitHub issue
opens via the same pattern the weekly workflow already uses for failures (see
"Workflow"). Three models are tried before that happens, so this is not fragile, but a
bad day means silence rather than a degraded page — worth being explicit about since it
is a different contract than the rest of the codebase.

A deeper grounding/quality check in the style of `quality.py` (truncation, number and
entity grounding, etc.) is a plausible future addition if low-quality-but-parseable
briefs turn out to be a real problem in practice. Not built preemptively.

## Storage

Mirrors the weekly digest's layout exactly, at a parallel path:

- `data/ai/latest.json` — same top-level shape as the weekly edition dict
  (`generated_at`, `intro`, `stories`), with `week` replaced by `date` (a `YYYY-MM-DD`
  string) and each story entry carrying `theme` / `what_happened` / `why_it_matters` /
  `watch_next` instead of `summary` / `why`.
- `data/ai/archive/<date>.json` — one file per published day, same shape.

`build_ai_edition()` and `save_ai_edition()` in `ai_daily.py` are the daily analogues
of `pipeline.build_edition()` / `pipeline.save_edition()`.

## Site rendering

`templates/ai_daily.html.jinja`: a variant of `edition.html.jinja` sharing the same
masthead, typography, and hairline-rule visual language (this is one publication with
two sections, not a different product), with:

- Tagline changed from "Published every Monday" to "Published daily · AI only"
- A theme badge (from the seven-value enum) replacing the general category pill
- Two callouts per story instead of one: the existing "WHY IT MATTERS" style box,
  plus a new "WHAT TO WATCH NEXT" box in the same visual treatment

`generate_ai_site()` in `ai_daily.py` mirrors `pipeline.generate_site()`'s
`_render_page` structure: renders `public/ai/index.html` for the latest edition and
`public/ai/archive/<date>.html` for every archived one.

`edition.html.jinja` gets one small, additive change: a quiet link to `/ai/` in the
header or footer.

## Syndication reuse

Rather than duplicating feed/sitemap logic, one small, backward-compatible
generalization to `syndication.py` (default arguments reproduce today's weekly output
unchanged):

`write_feed(editions, output_dir, site_url, limit=DEFAULT_FEED_LIMIT, title=SITE_TITLE, subtitle=SITE_SUBTITLE, entry_title=lambda week: f"Week ending {week}")`.
AI Daily calls it with its own `title`/`subtitle` and
`entry_title=lambda date: f"AI Daily — {date}"`, `output_dir=public/ai`,
`site_url=f"{site_url}/ai"`. Because `_page_url`/`_entry_id`/`_feed_url` all derive
from the `site_url` argument already, passing an `/ai`-suffixed base is sufficient —
no changes needed to those helpers. This alone produces a correct, independent
`public/ai/feed.xml`.

AI Daily also calls the existing `write_sitemap(editions, output_dir, site_url)`
unmodified, a second time, with `output_dir=public/ai`, `site_url=f"{site_url}/ai"`,
producing its own `public/ai/sitemap.xml`.

**Amendment from the design's original draft**: `write_robots` gaining an
`extra_sitemaps` parameter (to cross-reference `ai/sitemap.xml` from the one site-wide
`robots.txt`) turned out to need `pipeline.generate_site()` to know the AI Daily
sitemap's path — a small but real coupling from the weekly pipeline to a feature it has
no other reason to know about, for a purely cosmetic SEO completeness gain. Crawlers
already reach every `/ai/` page through ordinary link-following from the header/footer
link (`edition.html.jinja`'s new link, and `ai_daily.html.jinja`'s own archive nav), so
a sitemap cross-reference is not load-bearing for discoverability. Dropped: `write_robots`
stays untouched, and `public/robots.txt` continues to reference only `sitemap.xml`.
`public/ai/sitemap.xml` still exists and is directly submittable to a search console if
ever wanted — it's just not auto-referenced from the shared `robots.txt`.

**Field-name note**: `write_feed`/`write_sitemap` (and, on the weekly side,
`drop_recently_published`) all key off an edition dict's `"week"` field internally.
Rather than generalizing that key name throughout shared code, AI Daily's own edition
dicts use the clearer `"date"` field, and a small private helper in `ai_daily.py`
(`_for_shared_helpers`) builds a translated view (`{**edition, "week": edition["date"]}`)
at the three call sites that need it (`drop_recently_published`, `write_feed`,
`write_sitemap`). This keeps every shared function's signature and internals completely
untouched beyond the two changes already listed (`write_feed`'s new params, and
`drop_recently_published`'s `lookback: timedelta` rename).

## CLI

A new `ai-daily` Typer subcommand in `main.py`, structurally parallel to the existing
`run` command: `--data-dir` (default `data`), `--site-dir` (default `public`),
`--config`, `--dry-run`, `--force` (rebuild an already-published day). No `--skip-gate`
equivalent — there is no gate to skip; a `Briefer` call that exhausts its fallback chain
simply exits non-zero (see "Briefing step: Validation and failure handling").

**Consequence for the existing `run` command**: Typer only lets a single registered
command be invoked without naming it; the moment a second command (`ai-daily`) exists,
the CLI requires an explicit subcommand name for both. This means `uv run
lastweekintech --dry-run` becomes `uv run lastweekintech run --dry-run` going forward —
a real, if mechanical, change to the documented weekly invocation. Three places need
updating alongside `main.py`: `.github/workflows/main.yml`'s pipeline step (add `run`
after `lastweekintech`), `CLAUDE.md`'s Commands section, and every existing
`tests/test_cli.py` invocation (its shared `invoke()` helper is the only place this
needs to change; individual tests are unaffected).

## Workflow

New `.github/workflows/ai-daily.yml`, structured like the existing `main.yml`:

- `on: schedule: "0 13 * * *"` (13:00 UTC daily, clear of the Monday 08:00 UTC weekly
  run) plus `workflow_dispatch`
- `concurrency: group: ai-daily-digest` (independent of the weekly digest's
  `weekly-digest` group — the two never need to block each other, and their schedules
  never coincide)
- Same two-job shape as `main.yml`: `build-brief` (`permissions: contents: write`) then
  `notify-failure` (`permissions: issues: write`, `if: failure()`), with its own issue
  label (`ai-daily-failure`) so it does not collide with the weekly digest's
  `digest-failure` tracking issue
- The commit step stages `data` and `public` broadly, same pattern as the weekly
  workflow, which correctly picks up `data/ai` and `public/ai` (this run never touches
  the site-wide `public/sitemap.xml` / `public/robots.txt` — see the "Syndication
  reuse" amendment above)

Two existing files need small, additive updates:

- **`deploy-pages.yml`**: add `"AI Daily Briefing"` to its `workflow_run.workflows`
  list. Bot commits never fire `push` events, which is exactly why the weekly workflow
  needed this same hook — the daily one needs it for the same reason.
- **`tools/check_models.py`**: include `config.ai_daily.briefing.model_name` and
  `config.ai_daily.briefing.fallback_models` in the models checked against the live
  OpenRouter catalogue, alongside the existing summarizer/editor models.

## Testing

All new tests stay network-free, following the existing suite's conventions
(`tests/conftest.py` fixtures, `pythonpath = ["tests"]`, fixed `NOW` timestamp):

- `Briefer` unit tests via an injected `complete` fn: valid response, malformed JSON,
  wrong pick count, missing/invalid `theme`, refusal-shaped prose — mirrors
  `test_editor.py`'s structure
- Derived-`Config` construction (`dataclasses.replace` for `window_days`/`hn`/
  `weights`) verified through the existing injectable `parse`/`download`/`hn_fetch`
  fakes, confirming the daily pipeline actually uses the overridden values
- `drop_recently_published`'s `lookback: timedelta` signature change: update existing
  weekly-path tests for the new parameter shape, add a days-based case for the daily
  path
- `syndication.py` generalizations: a regression test asserting default arguments
  reproduce today's weekly `feed.xml`/`sitemap.xml` output unchanged, plus new tests
  for the `/ai` paths (title/subtitle/entry_title overrides, second sitemap, the
  combined `robots.txt`)
- `ai_daily.html.jinja` render test (structure, escaping, theme badge, two callouts),
  mirroring `test_site.py`
- CLI smoke test for the `ai-daily` subcommand, mirroring `test_cli.py`
- One end-to-end `build_ai_daily()` test with full fakes (parse/download/hn_fetch/
  complete), asserting correct `theme`/field population on the resulting `AiBrief`
  list and that the "recently covered" context passed to the prompt is built correctly
  from prior archive entries

## Explicitly out of scope for v1

- Folding the Perplexity consensus check into daily scoring (noted as a future
  option if candidate-pool recall turns out to be a problem in practice)
- A `quality.py`-style deterministic grounding check on briefs
- Any new RSS/blog sources beyond the existing 14 feeds + Hacker News
- A combined weekly+daily sitemap or feed (they stay fully independent artifacts)
- Any change to the weekly digest's own selection, scoring, or publishing behavior
  beyond the two additive changes noted above (the `drop_recently_published` signature,
  the small `/ai` link in `edition.html.jinja`)
