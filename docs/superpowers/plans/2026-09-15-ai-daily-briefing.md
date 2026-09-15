# AI Daily Briefing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, independent daily pipeline that finds the 4-5 most important AI news developments from the last 24 hours and publishes an analytical brief (what happened / why it matters / what to watch) for each, at `lastweekin.tech/ai/`.

**Architecture:** Reuses the weekly digest's fetch → dedupe → cluster → score → extract stages unmodified via a derived `Config`. Diverges after extraction: no mechanical category filter (it would miss chip/policy stories), and one LLM call (`Briefer`, in a new `briefer.py` mirroring `editor.py`) both selects the day's stories and writes the full analysis in one shot. A new `ai_daily.py` module owns everything else the daily pipeline needs that isn't shared with the weekly one: edition assembly, storage, site rendering, and its own feed/sitemap.

**Tech Stack:** Python 3.12, Typer CLI, Jinja2 templates, OpenAI SDK against OpenRouter, pytest. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-09-15-ai-daily-briefing-design.md](../specs/2026-09-15-ai-daily-briefing-design.md)

## Global Constraints

- Tooling: `uv` only, never Poetry. `ruff format` / `ruff check --fix` only, never Black/Flake8. New dependencies go in `pyproject.toml` via `uv add` — this plan adds none.
- Every new function that hits the network takes an injectable parameter (`parse`, `download`, `hn_fetch`, `complete`), exactly like the weekly pipeline, so the test suite stays network-free.
- No new `quality.py`/`validation.py`-style subsystem. A `Briefer` response is either fully valid (right count, every field populated, source cap respected) or the call is treated as failed and the fallback chain tries the next model.
- `Briefer` failing its entire fallback chain fails the whole day's run closed (exit code 1, nothing published) — there is no mechanical fallback that can write analysis prose.
- Every new/changed public function needs a docstring in the same terse, reasoning-carrying style already used throughout (see any existing docstring in `pipeline.py` or `editor.py` for the tone: state the non-obvious *why*, not the *what*).
- Run `uv run pytest` after every task; all prior tests must keep passing, not just the new ones.

---

### Task 1: Config — `ai_daily` section

**Files:**
- Modify: `src/lastweekintech/config.py`
- Modify: `src/lastweekintech/config.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `AiBriefingSettings` (fields: `model_name: str`, `fallback_models: list[str]`, `max_tokens: int`, `temperature: float`, `excerpt_chars: int`), `AiDailySettings` (fields: `enabled: bool`, `window_days: int`, `candidate_pool: int`, `story_count: int`, `min_story_count: int`, `repeat_lookback_days: int`, `max_per_source: int`, `hn: HNSettings`, `weights: Weights`, `briefing: AiBriefingSettings`), and `Config.ai_daily: AiDailySettings`. Every later task reads these exact names.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py` (append inside `TestFromYaml`, and a new `TestAiDaily` class after it):

```python
    def test_applies_ai_daily_defaults_when_the_section_is_omitted(self, tmp_path):
        config = Config.from_yaml(write_config(tmp_path, MINIMAL))
        assert config.ai_daily.enabled is True
        assert config.ai_daily.window_days == 1
        assert config.ai_daily.story_count == 5
        assert config.ai_daily.min_story_count == 4
        assert config.ai_daily.briefing.model_name == "anthropic/claude-sonnet-5"

    def test_rejects_an_unknown_ai_daily_key(self, tmp_path):
        body = MINIMAL + "\nai_daily:\n  nonsense: 1\n"
        with pytest.raises(ConfigError, match="Invalid configuration"):
            Config.from_yaml(write_config(tmp_path, body))


class TestAiDaily:
    def test_overrides_nested_hn_weights_and_briefing_settings(self, tmp_path):
        body = MINIMAL + (
            "\nai_daily:\n"
            "  story_count: 3\n"
            "  hn:\n"
            "    min_points: 5\n"
            "  weights:\n"
            "    hn: 10\n"
            "  briefing:\n"
            "    model_name: vendor/other-model\n"
            "    max_tokens: 1234\n"
        )
        config = Config.from_yaml(write_config(tmp_path, body))
        assert config.ai_daily.story_count == 3
        assert config.ai_daily.hn.min_points == 5
        # Untouched nested defaults still apply.
        assert config.ai_daily.hn.points_cap == 300
        assert config.ai_daily.weights.hn == 10
        assert config.ai_daily.briefing.model_name == "vendor/other-model"
        assert config.ai_daily.briefing.max_tokens == 1234
        assert config.ai_daily.briefing.temperature == 0.3

    def test_rejects_a_min_story_count_above_story_count(self, tmp_path):
        body = MINIMAL + "\nai_daily:\n  story_count: 3\n  min_story_count: 4\n"
        with pytest.raises(ConfigError, match="min_story_count"):
            Config.from_yaml(write_config(tmp_path, body))

    def test_rejects_a_candidate_pool_smaller_than_story_count(self, tmp_path):
        body = MINIMAL + "\nai_daily:\n  story_count: 5\n  candidate_pool: 3\n"
        with pytest.raises(ConfigError, match="candidate_pool"):
            Config.from_yaml(write_config(tmp_path, body))

    def test_rejects_an_empty_briefing_model_name(self, tmp_path):
        body = MINIMAL + '\nai_daily:\n  briefing:\n    model_name: ""\n'
        with pytest.raises(ConfigError, match="briefing.model_name"):
            Config.from_yaml(write_config(tmp_path, body))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `AttributeError: 'Config' object has no attribute 'ai_daily'` (and similar) for every new test.

- [ ] **Step 3: Add the settings dataclasses**

In `src/lastweekintech/config.py`, add after `EditorSettings` (which ends just before `PerplexitySettings`):

```python
@dataclass
class AiBriefingSettings:
    """The AI Daily analysis stage: one model call that selects and writes the brief."""

    model_name: str = "anthropic/claude-sonnet-5"
    fallback_models: list[str] = field(
        default_factory=lambda: ["anthropic/claude-haiku-4.5", "google/gemini-3.7-flash"]
    )
    # Five stories' worth of analysis in one JSON reply needs real headroom,
    # the same reasoning as editor.max_tokens.
    max_tokens: int = 6000
    temperature: float = 0.3
    # How many characters of each candidate's body the briefer reads.
    excerpt_chars: int = 600


@dataclass
class AiDailySettings:
    """The daily AI-only briefing: a second, independent pipeline.

    Reuses the weekly digest's fetch/dedupe/cluster/score/extract stages via
    a derived Config carrying this section's own window/hn/weights — see
    ``ai_daily.build_ai_daily``.
    """

    enabled: bool = True
    window_days: int = 1
    candidate_pool: int = 30
    story_count: int = 5
    # A floor, not a quota: a thin news day publishes four, not a padded fifth.
    min_story_count: int = 4
    repeat_lookback_days: int = 3
    max_per_source: int = 2
    hn: HNSettings = field(
        default_factory=lambda: HNSettings(min_points=20, points_cap=300, limit=200)
    )
    weights: Weights = field(default_factory=Weights)
    briefing: AiBriefingSettings = field(default_factory=AiBriefingSettings)
```

- [ ] **Step 4: Wire the new section into `Config`**

In `src/lastweekintech/config.py`, add the field to `Config` (after `editor`):

```python
    editor: EditorSettings = field(default_factory=EditorSettings)
    ai_daily: AiDailySettings = field(default_factory=AiDailySettings)
```

Add `"ai_daily"` to the `unknown` set in `from_yaml`:

```python
        unknown = set(data) - {
            "feeds",
            "hn",
            "weights",
            "window_days",
            "summarizer",
            "digest",
            "perplexity",
            "editor",
            "ai_daily",
            "site_url",
        }
```

Add a small builder function (nested dataclasses need the same `X(**data.get("x", {}))` treatment `Config` itself uses, one level deeper) right before `Config.from_yaml`:

```python
def _build_ai_daily(data: dict[str, Any]) -> AiDailySettings:
    """Construct AiDailySettings, building its nested dataclass fields the
    same way Config.from_yaml builds its own — a dict for any of ``hn``,
    ``weights`` or ``briefing`` becomes the matching settings object."""
    kwargs = dict(data)
    if "hn" in kwargs:
        kwargs["hn"] = HNSettings(**kwargs["hn"])
    if "weights" in kwargs:
        kwargs["weights"] = Weights(**kwargs["weights"])
    if "briefing" in kwargs:
        kwargs["briefing"] = AiBriefingSettings(**kwargs["briefing"])
    return AiDailySettings(**kwargs)
```

In the `cls(...)` call inside `from_yaml`, add:

```python
                editor=EditorSettings(**data.get("editor", {})),
                ai_daily=_build_ai_daily(data.get("ai_daily", {})),
```

- [ ] **Step 5: Add validation**

In `Config.validate()`, after the `min_ai_stories` check, add:

```python
        if self.ai_daily.story_count < 1:
            problems.append("ai_daily.story_count must be at least 1")
        if self.ai_daily.min_story_count > self.ai_daily.story_count:
            problems.append("ai_daily.min_story_count cannot exceed ai_daily.story_count")
        if self.ai_daily.candidate_pool < self.ai_daily.story_count:
            problems.append("ai_daily.candidate_pool must be at least ai_daily.story_count")
        if not self.ai_daily.briefing.model_name:
            problems.append("ai_daily.briefing.model_name is required")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS, all tests including the new ones.

- [ ] **Step 7: Add the production config section**

Append to `src/lastweekintech/config.yaml`, after the `summarizer:` section (end of file):

```yaml

# The daily AI-only briefing: a second, independent pipeline. Same feeds and
# Hacker News signal as the weekly digest, but a 24-hour window, no
# mechanical category filter (a chip export-control story would be tagged
# Hardware or Policy, not AI, so relevance is judged by the briefing model
# itself against a broad candidate pool), and one model call that both
# selects the 4-5 stories and writes the analysis for each.
ai_daily:
  enabled: true
  window_days: 1
  candidate_pool: 30
  story_count: 5
  # A floor, not a quota: a thin news day publishes four, not a padded fifth.
  min_story_count: 4
  repeat_lookback_days: 3
  max_per_source: 2
  hn:
    # 24 hours of traction runs far lower than a week's worth.
    min_points: 20
    points_cap: 300
    limit: 200
  weights:
    hn: 5
    src: 3
    rec: 1
  briefing:
    model_name: "anthropic/claude-sonnet-5"
    fallback_models:
      - "anthropic/claude-haiku-4.5"
      - "google/gemini-3.7-flash"
    # Five stories' worth of analysis in one JSON reply needs real headroom.
    max_tokens: 6000
    temperature: 0.3
    excerpt_chars: 600
```

- [ ] **Step 8: Confirm the shipped config is valid**

Run: `uv run python -c "from lastweekintech.config import get_config; c = get_config(); print(c.ai_daily)"`
Expected: prints the `AiDailySettings` object with no error.

- [ ] **Step 9: Run the full config test file once more, then commit**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

```bash
git add src/lastweekintech/config.py src/lastweekintech/config.yaml tests/test_config.py
git commit -m "feat: add ai_daily configuration section"
```

---

### Task 2: `drop_recently_published` — `lookback: timedelta`

**Files:**
- Modify: `src/lastweekintech/pipeline.py`
- Test: `tests/test_curation.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `drop_recently_published(stories, editions, now, lookback: timedelta, keep_at_least, date_field: str = "week")`. Task 5 calls this with `lookback=timedelta(days=...)` and the default `date_field`.

This is a pure rename plus one small additive parameter — `lookback_weeks: int` (always converted to `timedelta(weeks=lookback_weeks)` one line later) becomes `lookback: timedelta` directly, and a `date_field` parameter (default `"week"`, unused by any existing call site) lets a future caller read a differently-named date key without this function needing to know about "AI Daily" at all. Task 5's `ai_daily.py` in fact does **not** use `date_field` — it translates its own edition dicts to use the `"week"` key before calling this function (see Task 5's `_for_shared_helpers`), so `date_field` stays at its default everywhere in this codebase. It is included anyway because threading `lookback` through as a bare `timedelta` already removes the awkward "weeks" unit assumption baked into the old name; `date_field` costs nothing extra to add at the same time and documents that the "week" key is a naming choice, not a hard requirement, without forcing every caller to prove it.

- [ ] **Step 1: Update the failing tests**

In `tests/test_curation.py`, update `TestDropRecentlyPublished.drop`:

```python
    def drop(self, stories, editions, keep_at_least=1, lookback_weeks=3):
        return pipeline.drop_recently_published(
            stories,
            editions,
            now=NOW,
            lookback=timedelta(weeks=lookback_weeks),
            keep_at_least=keep_at_least,
        )
```

Add `from datetime import timedelta` to the top of `tests/test_curation.py` if not already imported (check first — run `grep -n "^from datetime\|^import datetime" tests/test_curation.py`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_curation.py -k TestDropRecentlyPublished -v`
Expected: FAIL — `TypeError: drop_recently_published() got an unexpected keyword argument 'lookback'`

- [ ] **Step 3: Update `drop_recently_published` and `_recently_published`**

In `src/lastweekintech/pipeline.py`, replace the signature and body of `drop_recently_published`:

```python
def drop_recently_published(
    stories: list[Story],
    editions: list[dict[str, Any]],
    now: datetime,
    lookback: timedelta,
    keep_at_least: int,
    date_field: str = "week",
) -> list[Story]:
    """Remove stories the recent archive already published.

    Ranking is driven by Hacker News traction, which persists for days, so a
    story hot for eight days can top two consecutive editions. Under the old
    recency-driven ranking that never happened — zero repeats across 44
    archived editions — so this is reasoned from the new ranking rather than
    fitted to history.

    Both identity tests are used because repeats arrive both ways: the same
    canonical URL resurfacing on Hacker News, and a second outlet's write-up of
    a story we already ran. Everything here degrades to "publish it" on bad
    input; a broken archive must not be able to empty an edition. Nor may a
    week of repeats: ``keep_at_least`` restores the highest-ranked repeats until
    the digest can be filled, so the filter can only ever reorder a starved
    week, never shorten it.

    ``date_field`` names the key each edition dict's date lives under —
    "week" for the weekly digest's archive. A caller with a differently-shaped
    archive (a daily one, say) passes its own field name rather than this
    function needing to know anything about that pipeline.
    """
    if lookback <= timedelta(0) or not stories:
        return stories

    urls, titles = _recently_published(editions, now, lookback, date_field)
    if not urls and not titles:
        return stories

    fresh, repeats = [], []
    for story in stories:
        if _was_published(story, urls, titles):
            repeats.append(story)
        else:
            fresh.append(story)

    if not repeats:
        return stories

    shortfall = keep_at_least - len(fresh)
    restored = repeats[:shortfall] if shortfall > 0 else []
    if restored:
        logging.info(f"Restored {len(restored)} repeats to keep the edition full.")
    logging.info(f"Dropped {len(repeats) - len(restored)} stories published in a recent edition.")

    # Rebuild in the caller's order so the ranking survives the filter.
    kept = {id(s) for s in fresh} | {id(s) for s in restored}
    return [s for s in stories if id(s) in kept]


def _recently_published(
    editions: list[dict[str, Any]],
    now: datetime,
    lookback: timedelta,
    date_field: str = "week",
) -> tuple[set[str], list[str]]:
    """Collect the URLs and titles published within the lookback window."""
    cutoff = now.date() - lookback
    urls: set[str] = set()
    titles: list[str] = []

    for edition in editions:
        if not isinstance(edition, dict):
            continue
        published = _edition_date(edition.get(date_field))
        if published is None or published < cutoff:
            continue
        for entry in edition.get("stories") or []:
            if not isinstance(entry, dict):
                continue
            key = normalize_url(str(entry.get("url") or ""))
            if key:
                urls.add(key)
            title = entry.get("title")
            if isinstance(title, str) and title.strip():
                titles.append(title)

    return urls, titles
```

Note: the old guard was `if lookback_weeks < 1`; with a `timedelta` the equivalent is `if lookback <= timedelta(0)`.

- [ ] **Step 4: Update the one call site in `build_digest`**

In `src/lastweekintech/pipeline.py`, inside `build_digest`, change:

```python
        ranked = drop_recently_published(
            stories,
            editions or [],
            now=now,
            lookback_weeks=config.digest.repeat_lookback_weeks,
            keep_at_least=config.digest.story_count,
        )
```

to:

```python
        ranked = drop_recently_published(
            stories,
            editions or [],
            now=now,
            lookback=timedelta(weeks=config.digest.repeat_lookback_weeks),
            keep_at_least=config.digest.story_count,
        )
```

- [ ] **Step 5: Update remaining `test_curation.py` call sites**

Every other test in `TestDropRecentlyPublished` calls `self.drop(...)` (already fixed in Step 1) except none pass `lookback_weeks` as a positional argument, so no further changes are needed there. Search for any other direct `pipeline.drop_recently_published(` calls outside this class:

Run: `grep -rn "drop_recently_published(" tests/`
Expected: only the one call site inside `TestDropRecentlyPublished.drop`.

- [ ] **Step 6: Run the full test suite to verify it passes**

Run: `uv run pytest -q`
Expected: PASS, all tests.

- [ ] **Step 7: Commit**

```bash
git add src/lastweekintech/pipeline.py tests/test_curation.py
git commit -m "refactor: drop_recently_published takes a timedelta lookback, not weeks"
```

---

### Task 3: `syndication.write_feed` — title/subtitle/entry_title

**Files:**
- Modify: `src/lastweekintech/syndication.py`
- Test: `tests/test_syndication.py`

**Interfaces:**
- Produces: `write_feed(editions, output_dir, site_url, limit=DEFAULT_FEED_LIMIT, title=SITE_TITLE, subtitle=SITE_SUBTITLE, entry_title: Callable[[str], str] = <default>)`. Task 6 calls this with AI Daily's own title/subtitle/entry_title.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_syndication.py`, inside `TestWriteFeed`:

```python
    def test_accepts_a_custom_title_and_subtitle(self, tmp_path):
        root = parse(
            syndication.write_feed(
                [edition()], tmp_path, SITE, title="Custom Title", subtitle="Custom subtitle"
            )
        )
        assert root.findtext(f"{ATOM}title") == "Custom Title"
        assert root.findtext(f"{ATOM}subtitle") == "Custom subtitle"

    def test_defaults_reproduce_the_weekly_title_and_subtitle(self, tmp_path):
        root = parse(syndication.write_feed([edition()], tmp_path, SITE))
        assert root.findtext(f"{ATOM}title") == syndication.SITE_TITLE
        assert root.findtext(f"{ATOM}subtitle") == syndication.SITE_SUBTITLE

    def test_accepts_a_custom_entry_title(self, tmp_path):
        root = parse(
            syndication.write_feed(
                [edition()], tmp_path, SITE, entry_title=lambda week: f"AI Daily — {week}"
            )
        )
        entry_title = root.find(f"{ATOM}entry/{ATOM}title")
        assert entry_title.text == "AI Daily — 2026-08-10"

    def test_default_entry_title_reproduces_the_weekly_wording(self, tmp_path):
        root = parse(syndication.write_feed([edition()], tmp_path, SITE))
        entry_title = root.find(f"{ATOM}entry/{ATOM}title")
        assert entry_title.text == "Week ending 2026-08-10"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_syndication.py -k "custom_title or custom_entry_title" -v`
Expected: FAIL — `TypeError: write_feed() got an unexpected keyword argument 'title'`

- [ ] **Step 3: Add the parameters**

In `src/lastweekintech/syndication.py`, add the import:

```python
from collections.abc import Callable
```

(add alongside the existing `import re` / `import html` block at the top).

Replace `write_feed` and `_append_entry`:

```python
def write_feed(
    editions: list[dict[str, Any]],
    output_dir: Path,
    site_url: str,
    limit: int = DEFAULT_FEED_LIMIT,
    title: str = SITE_TITLE,
    subtitle: str = SITE_SUBTITLE,
    entry_title: Callable[[str], str] = lambda week: f"Week ending {week}",
) -> Path:
    """Write an Atom 1.0 feed with one entry per edition, newest first.

    The entry is the edition rather than the story: this is a weekly digest, and a
    reader subscribes to the week. Seven entries a week would also make every
    edition look like seven unrelated updates in a reader's timeline.

    ``title``/``subtitle``/``entry_title`` default to the weekly digest's own
    wording, so every existing call site is unaffected; a second feed for a
    different section of the site (see ``ai_daily.generate_ai_site``) passes
    its own.
    """
    base = _site_base(site_url)
    ordered = _newest_first(editions)[: max(limit, 0)]

    root = ET.Element("feed", {"xmlns": ATOM_NS})
    _text(root, "title", title)
    _text(root, "subtitle", subtitle)
    _text(root, "id", f"{base}/")
    _text(root, "updated", _updated(ordered[0]) if ordered else _EPOCH)
    ET.SubElement(
        root, "link", {"rel": "self", "type": "application/atom+xml", "href": _feed_url(base)}
    )
    ET.SubElement(root, "link", {"rel": "alternate", "type": "text/html", "href": f"{base}/"})
    _text(ET.SubElement(root, "author"), "name", SITE_AUTHOR)

    for edition in ordered:
        _append_entry(root, edition, base, entry_title)

    return _write_xml(root, output_dir / FEED_FILENAME)
```

```python
def _append_entry(
    root: ET.Element, edition: dict[str, Any], base: str, entry_title: Callable[[str], str]
) -> None:
    week = _week(edition)
    page = _page_url(base, edition)

    entry = ET.SubElement(root, "entry")
    _text(entry, "title", entry_title(week))
    _text(entry, "id", _entry_id(base, week))
    _text(entry, "updated", _updated(edition))
    ET.SubElement(entry, "link", {"rel": "alternate", "type": "text/html", "href": page})
    # type="html" means the value is an escaped HTML fragment, which is what a
    # text node gives us for free: the serializer escapes it on the way out.
    _text(entry, "content", _stories_html(edition), attrib={"type": "html"})
```

`write_syndication` calls `write_feed(editions, output_dir, site_url, limit=limit)` with no other args, so it needs no change — it keeps using the defaults.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_syndication.py -v`
Expected: PASS, all tests including the four new ones.

- [ ] **Step 5: Run the full suite, then commit**

Run: `uv run pytest -q`
Expected: PASS

```bash
git add src/lastweekintech/syndication.py tests/test_syndication.py
git commit -m "feat: syndication.write_feed accepts a title, subtitle and entry title"
```

---

### Task 4: `briefer.py` — the selection-and-analysis model call

**Files:**
- Create: `src/lastweekintech/briefer.py`
- Test: `tests/test_briefer.py`

**Interfaces:**
- Consumes: `Story` (from `lastweekintech.domain`), `CompleteFn`/`Completion`/`OPENROUTER_BASE_URL` (from `lastweekintech.summarizer`), `AiBriefingSettings` (from `lastweekintech.config`, Task 1).
- Produces: `THEMES: tuple[str, ...]`, `BriefPick` (fields: `n: int`, `theme: str`, `what_happened: str`, `why_it_matters: str`, `watch_next: str`), `BriefVerdict` (fields: `picks: list[BriefPick]`, `intro: str`), `Briefer` class with `.brief(candidates: list[Story], count: int, min_count: int, max_per_source: int, recent_titles: list[str]) -> BriefVerdict | None`, `.last_model: str | None`, `.max_tokens: int`, `.last_completion_tokens: int | None`, `.last_reasoning_tokens: int | None`. Task 5 imports `Briefer`, `BriefVerdict`, `BriefPick` from this module.

This mirrors `editor.py`'s `Editor`/`EditorVerdict`/`Pick` almost exactly (same `CompleteFn` injection, same fallback-across-models loop, same "trustworthy or bust" validation philosophy), with three differences: the response carries a full analysis per pick instead of a one-line "why"; the pick count is a range (`min_count` to `count`) instead of an exact number; and a per-source cap is enforced as a second validation gate (reject the whole verdict rather than trying to mechanically fix it, since there's no fallback content to substitute a rejected pick with).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_briefer.py`:

```python
"""Tests for AI Daily's briefing step: the Briefer and the guards around its verdict."""

import json

import pytest
from conftest import make_article, make_story

from lastweekintech.briefer import THEMES, Briefer, BriefVerdict
from lastweekintech.config import AiBriefingSettings
from lastweekintech.summarizer import Completion


def canned(payload, finish_reason=None):
    """A complete() that always answers with ``payload`` (dict → JSON)."""
    text = json.dumps(payload) if isinstance(payload, dict) else payload

    def complete(model, messages, max_tokens):
        return Completion(text=text, finish_reason=finish_reason)

    return complete


def make_briefer(payload, **settings):
    return Briefer(AiBriefingSettings(**settings), complete=canned(payload))


def pick(n, theme=THEMES[0]):
    return {
        "n": n,
        "theme": theme,
        "what_happened": f"Something happened in pick {n}.",
        "why_it_matters": f"Pick {n} matters because of X.",
        "watch_next": f"Watch for Y after pick {n}.",
    }


def verdict_for(*ns, intro="A big day for AI."):
    return {"intro": intro, "picks": [pick(n) for n in ns]}


def pool(count=6, source="Example"):
    return [
        make_story(
            title=f"story {i}",
            score=100 - i,
            articles=[make_article(title=f"story {i}", url=f"https://x/{i}", source=source)],
        )
        for i in range(count)
    ]


class TestBrief:
    def test_returns_the_picks_and_intro(self):
        briefer = make_briefer(verdict_for(2, 1, 3))
        verdict = briefer.brief(pool(), count=3, min_count=3, max_per_source=0, recent_titles=[])
        assert verdict is not None
        assert [p.n for p in verdict.picks] == [2, 1, 3]
        assert verdict.picks[0].theme == THEMES[0]
        assert verdict.picks[0].what_happened == "Something happened in pick 2."
        assert verdict.intro == "A big day for AI."

    def test_accepts_a_pick_count_within_the_min_max_range(self):
        briefer = make_briefer(verdict_for(1, 2, 3, 4))
        verdict = briefer.brief(pool(), count=5, min_count=4, max_per_source=0, recent_titles=[])
        assert verdict is not None
        assert len(verdict.picks) == 4

    def test_rejects_a_pick_count_below_the_minimum(self):
        briefer = make_briefer(verdict_for(1, 2, 3))
        verdict = briefer.brief(pool(), count=5, min_count=4, max_per_source=0, recent_titles=[])
        assert verdict is None

    def test_records_which_model_answered(self):
        briefer = make_briefer(verdict_for(1, 2), model_name="test/briefer")
        briefer.brief(pool(), count=2, min_count=2, max_per_source=0, recent_titles=[])
        assert briefer.last_model == "test/briefer"

    def test_records_the_token_usage_of_the_answering_model(self):
        def complete(model, messages, max_tokens):
            return Completion(
                text=json.dumps(verdict_for(1, 2)),
                completion_tokens=5100,
                reasoning_tokens=4800,
            )

        briefer = Briefer(AiBriefingSettings(), complete=complete)
        briefer.brief(pool(), count=2, min_count=2, max_per_source=0, recent_titles=[])
        assert briefer.last_completion_tokens == 5100
        assert briefer.last_reasoning_tokens == 4800

    def test_tolerates_prose_around_the_json(self):
        answer = f"Here is the briefing:\n```json\n{json.dumps(verdict_for(1, 2))}\n```"
        briefer = make_briefer(answer)
        verdict = briefer.brief(pool(), count=2, min_count=2, max_per_source=0, recent_titles=[])
        assert verdict is not None

    @pytest.mark.parametrize(
        "payload",
        [
            "not json at all",
            {"intro": "x", "picks": "not a list"},
            verdict_for(1),  # below min_count=2
            verdict_for(1, 1),  # duplicate pick
            verdict_for(1, 99),  # out of range
            {"picks": [{"n": 1, "theme": "Not A Real Theme", "what_happened": "x",
                        "why_it_matters": "y", "watch_next": "z"}, pick(2)]},
            {"picks": [{"n": 1, "theme": THEMES[0], "what_happened": "",
                        "why_it_matters": "y", "watch_next": "z"}, pick(2)]},
        ],
    )
    def test_an_unusable_verdict_yields_none(self, payload):
        briefer = make_briefer(payload)
        assert briefer.brief(
            pool(), count=2, min_count=2, max_per_source=0, recent_titles=[]
        ) is None

    def test_rejects_a_verdict_that_breaks_the_source_cap(self):
        # Every candidate shares the same source, so a cap of 1 with 2 picks
        # can never be satisfied.
        briefer = make_briefer(verdict_for(1, 2))
        verdict = briefer.brief(
            pool(source="OneOutlet"), count=2, min_count=2, max_per_source=1, recent_titles=[]
        )
        assert verdict is None

    def test_a_zero_max_per_source_disables_the_cap(self):
        briefer = make_briefer(verdict_for(1, 2))
        verdict = briefer.brief(
            pool(source="OneOutlet"), count=2, min_count=2, max_per_source=0, recent_titles=[]
        )
        assert verdict is not None

    def test_falls_back_across_models(self):
        answers = iter([RuntimeError("down"), json.dumps(verdict_for(1, 2))])

        def complete(model, messages, max_tokens):
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return Completion(text=answer)

        briefer = Briefer(
            AiBriefingSettings(model_name="a/primary", fallback_models=["b/backup"]),
            complete=complete,
        )
        verdict = briefer.brief(pool(), count=2, min_count=2, max_per_source=0, recent_titles=[])
        assert verdict is not None
        assert briefer.last_model == "b/backup"

    def test_an_empty_pool_yields_none_without_a_model_call(self):
        def complete(model, messages, max_tokens):
            raise AssertionError("should not be called")

        briefer = Briefer(AiBriefingSettings(), complete=complete)
        assert briefer.brief([], count=3, min_count=2, max_per_source=0, recent_titles=[]) is None

    def test_the_prompt_lists_recent_titles_to_avoid(self):
        seen = {}

        def complete(model, messages, max_tokens):
            seen["prompt"] = messages[0]["content"]
            return Completion(text=json.dumps(verdict_for(1, 2)))

        Briefer(AiBriefingSettings(), complete=complete).brief(
            pool(), count=2, min_count=2, max_per_source=0,
            recent_titles=["Yesterday's big story"],
        )
        assert "Yesterday's big story" in seen["prompt"]

    def test_the_prompt_shows_signals_and_marks_missing_bodies(self):
        seen = {}

        def complete(model, messages, max_tokens):
            seen["prompt"] = messages[1]["content"]
            return Completion(text=json.dumps(verdict_for(1, 2)))

        rich = make_story(
            title="Covered everywhere",
            score=5,
            articles=[make_article(title="Covered everywhere", hn_points=400)],
        )
        rich.consensus = True
        bare = make_story(
            title="Paywalled thing",
            score=4,
            articles=[make_article(title="Paywalled thing", content=None)],
        )
        Briefer(AiBriefingSettings(), complete=complete).brief(
            [rich, bare], count=2, min_count=2, max_per_source=0, recent_titles=[]
        )
        assert "400 HN points" in seen["prompt"]
        assert "press consensus" in seen["prompt"]
        assert "NO ARTICLE TEXT AVAILABLE" in seen["prompt"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_briefer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lastweekintech.briefer'`

- [ ] **Step 3: Write `briefer.py`**

Create `src/lastweekintech/briefer.py`:

```python
"""AI Daily's briefing step: one model call selects the day and writes it.

Unlike the weekly digest, where a mechanical ranking narrows the pool and an
editor call only reorders it before a separate summarizer writes each story,
AI Daily has no per-story analysis a model didn't write — there is nothing
mechanical that can substitute for "why this matters." So this call does
both jobs at once, and a verdict is either fully usable or it is discarded
and the next model in the chain is tried; there is no partial credit.
"""

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field

from openai import OpenAI

from lastweekintech.config import AiBriefingSettings
from lastweekintech.domain import Story
from lastweekintech.summarizer import OPENROUTER_BASE_URL, CompleteFn, Completion

THEMES = (
    "Capabilities",
    "Companies",
    "Open Source",
    "Policy & Regulation",
    "Security & Safety",
    "Infrastructure & Chips",
    "Industry & Economics",
)

SYSTEM_PROMPT = (
    "You are the AI editor for LastWeekIn.Tech's daily AI briefing. From the "
    "numbered candidates, choose the {min_count}-{max_count} most important AI "
    "news developments from the last 24 hours.\n"
    "\n"
    "Prioritize:\n"
    "- AI capabilities: genuine advances in what models can do.\n"
    "- Major companies: OpenAI, Anthropic, Google, Meta, Microsoft and similar "
    "moving the industry.\n"
    "- Open-source models: new releases, weights, or major community shifts.\n"
    "- Regulation and policy: laws, export controls, government action on AI.\n"
    "- Security and safety: incidents, vulnerabilities, safety research.\n"
    "- Infrastructure and chips: compute, data centers, chip supply and export "
    "controls.\n"
    "- Industry economics: funding, valuations, market shifts tied to AI.\n"
    "\n"
    "Downrank low-impact product announcements, minor feature releases and "
    "listicles. Skip a story that is a rehash of something already covered "
    "(see below) unless it represents a materially different development.\n"
    "\n"
    "{recent_section}"
    "At most {max_per_source} picks may share the same outlet.\n"
    "Never pick a candidate marked as having no article text: it cannot be "
    "analyzed.\n"
    "\n"
    "For each pick, write:\n"
    '- "theme": exactly one of {themes}\n'
    '- "what_happened": 2-4 factual sentences, using only facts in the article.\n'
    '- "why_it_matters": a paragraph on significance and near-term implications.\n'
    '- "watch_next": 1-2 sentences on what to watch for next.\n'
    "\n"
    "Reply with JSON only, no prose around it:\n"
    '{{"intro": "1-2 sentences on the day\'s overall AI news shape", '
    '"picks": [{{"n": <candidate number>, "theme": "<theme>", '
    '"what_happened": "<...>", "why_it_matters": "<...>", '
    '"watch_next": "<...>"}}]}}'
)


@dataclass
class BriefPick:
    """One AI Daily pick: a candidate number plus its full analysis."""

    n: int
    theme: str
    what_happened: str
    why_it_matters: str
    watch_next: str


@dataclass
class BriefVerdict:
    """The day's AI briefing: picks in print order, plus the day in brief."""

    picks: list[BriefPick] = field(default_factory=list)
    intro: str = ""


class Briefer:
    """Selects and writes the AI Daily brief, falling back across models."""

    def __init__(self, settings: AiBriefingSettings, complete: CompleteFn | None = None):
        self.settings = settings
        self.models = [settings.model_name, *settings.fallback_models]
        self.last_model: str | None = None
        self.max_tokens = settings.max_tokens
        self.last_completion_tokens: int | None = None
        self.last_reasoning_tokens: int | None = None
        self._complete = complete or self._complete_via_api

        if complete is None:
            if not os.getenv("OPENROUTER_API_KEY"):
                raise ValueError("OPENROUTER_API_KEY environment variable not set.")
            self._client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

    def brief(
        self,
        candidates: list[Story],
        count: int,
        min_count: int,
        max_per_source: int,
        recent_titles: list[str],
    ) -> BriefVerdict | None:
        """Pick and write the day's briefing; ``None`` means nothing usable came back.

        A verdict is only returned when a model produced between ``min_count``
        and ``count`` distinct, in-range picks, every field populated, every
        theme valid, and the per-source cap respected — anything less is not
        worth publishing, since there is no mechanical fallback that can write
        the analysis a rejected pick would have carried.
        """
        if not candidates:
            return None
        max_count = min(count, len(candidates))
        min_count = min(min_count, max_count)
        self.last_model = None
        self.last_completion_tokens = None
        self.last_reasoning_tokens = None

        recent_section = (
            "Recently covered — skip these unless there's a material update:\n"
            + "\n".join(f"- {title}" for title in recent_titles)
            + "\n\n"
            if recent_titles
            else ""
        )
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    min_count=min_count,
                    max_count=max_count,
                    recent_section=recent_section,
                    max_per_source=max_per_source,
                    themes=", ".join(f'"{theme}"' for theme in THEMES),
                ),
            },
            {"role": "user", "content": _render_candidates(candidates, self.settings)},
        ]

        for model in self.models:
            try:
                logging.info(f"Asking {model} for the AI Daily briefing...")
                completion = self._complete(model, messages, self.settings.max_tokens)
            except Exception as e:  # noqa: BLE001 - try the next model instead
                logging.warning(f"Briefing model {model} failed: {e}")
                continue

            verdict = _parse_verdict(
                completion.text, pool=len(candidates), min_count=min_count, max_count=max_count
            )
            if verdict and _within_source_cap(verdict, candidates, max_per_source):
                self.last_model = model
                self.last_completion_tokens = completion.completion_tokens
                self.last_reasoning_tokens = completion.reasoning_tokens
                if completion.completion_tokens is not None:
                    logging.info(
                        f"Briefer {model} spent {completion.completion_tokens} tokens "
                        f"({completion.reasoning_tokens} reasoning) of "
                        f"{self.max_tokens} budgeted."
                    )
                return verdict
            logging.warning(
                f"Briefing model {model} returned an unusable verdict "
                f"(finish_reason={completion.finish_reason!r}): {completion.text[:200]!r}"
            )

        logging.warning("No briefing model produced a usable verdict.")
        return None

    def _complete_via_api(
        self, model: str, messages: list[dict[str, str]], max_tokens: int
    ) -> Completion:
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=self.settings.temperature,
        )
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        return Completion(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            completion_tokens=getattr(usage, "completion_tokens", None),
            reasoning_tokens=getattr(details, "reasoning_tokens", None),
        )


def _render_candidates(candidates: list[Story], settings: AiBriefingSettings) -> str:
    lines = []
    for n, story in enumerate(candidates, start=1):
        outlets = sorted({a.source for a in story.articles})
        points = max((a.hn_points or 0) for a in story.articles)
        signals = [f"outlets: {', '.join(outlets)}"]
        if points:
            signals.append(f"{points} HN points")
        if story.consensus:
            signals.append("press consensus")

        body = max((a.content or "" for a in story.articles), key=len)
        excerpt = (
            " ".join(body.split())[: settings.excerpt_chars]
            if body
            else "NO ARTICLE TEXT AVAILABLE — cannot be analyzed"
        )
        lines.append(
            f"{n}. {story.title}\n   ({'; '.join(signals)})\n   {excerpt}"
        )
    return "Candidates:\n\n" + "\n\n".join(lines)


def _parse_verdict(answer: str, pool: int, min_count: int, max_count: int) -> BriefVerdict | None:
    """Read the model's JSON, accepting nothing less than a complete, valid briefing."""
    match = re.search(r"\{.*\}", answer or "", re.DOTALL)
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("picks"), list):
        return None

    picks = []
    seen = set()
    for entry in raw["picks"]:
        if not isinstance(entry, dict):
            return None
        n = entry.get("n")
        theme = entry.get("theme")
        what_happened = str(entry.get("what_happened") or "").strip()
        why_it_matters = str(entry.get("why_it_matters") or "").strip()
        watch_next = str(entry.get("watch_next") or "").strip()

        if not isinstance(n, int) or not 1 <= n <= pool or n in seen:
            return None
        if theme not in THEMES:
            return None
        if not what_happened or not why_it_matters or not watch_next:
            return None

        seen.add(n)
        picks.append(
            BriefPick(
                n=n,
                theme=theme,
                what_happened=what_happened,
                why_it_matters=why_it_matters,
                watch_next=watch_next,
            )
        )

    if not min_count <= len(picks) <= max_count:
        return None
    return BriefVerdict(picks=picks[:max_count], intro=str(raw.get("intro") or "").strip())


def _story_source(story: Story) -> str:
    """The outlet a pick counts against for the per-source cap.

    Mirrors ``pipeline._representative_article``'s preference (original
    reporting over an aggregator's rewrite, then traction, then body length),
    duplicated rather than imported so this module stays a dependency-free
    leaf like ``editor.py`` and ``summarizer.py``.
    """
    if not story.articles:
        return ""
    representative = max(
        story.articles,
        key=lambda a: (not a.aggregator, a.hn_points or 0, len(a.content or "")),
    )
    return representative.source


def _within_source_cap(verdict: BriefVerdict, candidates: list[Story], max_per_source: int) -> bool:
    if max_per_source < 1:
        return True
    counts = Counter(_story_source(candidates[pick.n - 1]) for pick in verdict.picks)
    return all(n <= max_per_source for n in counts.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_briefer.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Run the full suite, lint, and commit**

Run: `uv run pytest -q && uv run ruff check src/lastweekintech/briefer.py tests/test_briefer.py`
Expected: PASS, no lint errors.

```bash
git add src/lastweekintech/briefer.py tests/test_briefer.py
git commit -m "feat: add Briefer, the AI Daily selection-and-analysis model call"
```

---

### Task 5: `ai_daily.py` — orchestration, domain types, storage

**Files:**
- Modify: `src/lastweekintech/domain.py`
- Create: `src/lastweekintech/ai_daily.py`
- Test: `tests/test_ai_daily.py`

**Interfaces:**
- Consumes: `Briefer`, `BriefVerdict`, `BriefPick` (Task 4); `drop_recently_published` with `lookback`/`date_field` (Task 2); `syndication.write_feed` with `title`/`subtitle`/`entry_title` (Task 3); `Config`/`AiDailySettings` (Task 1); `fetch_articles`, `hn.fetch_hn_articles`, `dedupe_articles`, `cluster_articles`, `score_stories`, `extract_content`, `ARCHIVE_DIRNAME`, `PACKAGE_DIR` (all existing, unmodified, from `pipeline.py`).
- Produces: `AiBrief`, `AiDigest` (in `domain.py`); `build_ai_daily(config, briefer, now=None, parse=None, download=None, hn_fetch=None, delay=0.5, editions=None) -> AiDigest`; `build_ai_edition(briefs, date, now=None, intro=None) -> dict`; `save_ai_edition(edition, data_dir) -> tuple[Path, Path]`; `list_ai_editions(data_dir) -> list[dict]`; `AI_DAILY_SUBDIR: str`. Task 6 adds `generate_ai_site`/`_render_ai_page` to this same file. Task 7's CLI imports `ai_daily.build_ai_daily`, `ai_daily.build_ai_edition`, `ai_daily.save_ai_edition`, `ai_daily.list_ai_editions`, `ai_daily.AI_DAILY_SUBDIR`.

- [ ] **Step 1: Add the domain types**

In `src/lastweekintech/domain.py`, add after `Digest`:

```python
@dataclass
class AiBrief:
    """One AI Daily item: a story plus its analysis."""

    title: str
    articles: list[Article] = field(default_factory=list)
    theme: str = ""
    what_happened: str = ""
    why_it_matters: str = ""
    watch_next: str = ""
    score: float = 0.0


@dataclass
class AiDigest:
    """One day's AI briefing as the pipeline hands it to the publisher."""

    briefs: list[AiBrief] = field(default_factory=list)
    # The briefer's 1-2 sentence read on the day; None when every model in
    # its fallback chain failed, in which case ``briefs`` is also empty.
    intro: str | None = None
```

- [ ] **Step 2: Write the failing tests for `build_ai_daily` and edition storage**

Create `tests/test_ai_daily.py`:

```python
"""End-to-end tests for the AI Daily pipeline, with the network stubbed out."""

import json
from datetime import timedelta

from conftest import NOW, make_article, make_story
from test_build_digest import FakeEntry, entry, feeds_for

from lastweekintech import ai_daily
from lastweekintech.briefer import BriefPick, BriefVerdict
from lastweekintech.config import Feed


class FakeBriefer:
    def __init__(self, verdict):
        self.verdict = verdict
        self.last_model = "fake/briefer"
        self.last_completion_tokens = None
        self.last_reasoning_tokens = None
        self.max_tokens = 0
        self.seen_recent_titles = None

    def brief(self, candidates, count, min_count, max_per_source, recent_titles):
        self.seen_recent_titles = recent_titles
        self.seen_candidates = [s.title for s in candidates]
        return self.verdict


def small_config(config):
    config.ai_daily.story_count = 2
    config.ai_daily.min_story_count = 1
    config.ai_daily.candidate_pool = 10
    config.feeds = [Feed(name="Ars Technica", url="ars"), Feed(name="WIRED", url="wired")]
    return config


def brief_pick(n, theme="Capabilities"):
    return BriefPick(
        n=n,
        theme=theme,
        what_happened=f"Thing {n} happened.",
        why_it_matters=f"It matters because {n}.",
        watch_next=f"Watch for {n} next.",
    )


def unique_entries(count, prefix="https://ars.example"):
    subjects = [
        "OpenAI ships a new reasoning model",
        "Anthropic publishes a safety paper",
        "EU regulators open an AI antitrust probe",
        "A startup open-sources a 70B model",
        "Nvidia announces its next AI chip",
        "A major AI funding round closes",
    ]
    return [
        entry(f"{subjects[i % len(subjects)]} ({i})", f"{prefix}/{i}", age_hours=1 + i)
        for i in range(count)
    ]


def run(config, briefer, **overrides):
    kwargs = {
        "now": NOW,
        "parse": feeds_for({}),
        "download": lambda url: f"body for {url}",
        "hn_fetch": lambda url, params: {"hits": []},
        "delay": 0,
    }
    return ai_daily.build_ai_daily(config, briefer, **(kwargs | overrides))


class TestBuildAiDaily:
    def test_produces_the_briefer_verdicts_as_ai_briefs(self, config):
        parse = feeds_for({"ars": unique_entries(5), "wired": []})
        briefer = FakeBriefer(BriefVerdict(picks=[brief_pick(2), brief_pick(1)], intro="A day."))
        digest = run(small_config(config), briefer, parse=parse)
        assert digest.intro == "A day."
        assert len(digest.briefs) == 2
        assert digest.briefs[0].what_happened == "Thing 2 happened."
        assert digest.briefs[0].theme == "Capabilities"

    def test_a_failed_briefer_yields_an_empty_digest(self, config):
        parse = feeds_for({"ars": unique_entries(5), "wired": []})
        digest = run(small_config(config), FakeBriefer(None), parse=parse)
        assert digest.briefs == []
        assert digest.intro is None

    def test_uses_the_ai_daily_window_and_hn_settings_not_the_weekly_ones(self, config):
        # A story published 3 days ago is outside ai_daily's 1-day window but
        # would be inside the weekly config's 7-day window. An empty verdict
        # is used deliberately: with candidates correctly filtered down to
        # none, a non-empty canned verdict would make select_ai_daily_edition
        # index into an empty candidate list.
        config = small_config(config)
        config.window_days = 7
        parse = feeds_for({
            "ars": [entry("An old AI story", "https://ars.example/old", age_hours=72)],
            "wired": [],
        })
        briefer = FakeBriefer(BriefVerdict(picks=[]))
        run(config, briefer, parse=parse)
        assert briefer.seen_candidates == []

    def test_passes_recent_archive_titles_to_the_briefer(self, config):
        parse = feeds_for({"ars": unique_entries(3), "wired": []})
        briefer = FakeBriefer(BriefVerdict(picks=[brief_pick(1)]))
        editions = [{
            "date": (NOW - timedelta(days=1)).strftime("%Y-%m-%d"),
            "generated_at": NOW.isoformat(),
            "stories": [{"title": "Yesterday's AI story", "url": "https://x/1"}],
        }]
        run(small_config(config), briefer, parse=parse, editions=editions)
        assert "Yesterday's AI story" in briefer.seen_recent_titles

    def test_drops_a_story_already_published_in_the_ai_daily_archive(self, config):
        # story_count=1 (rather than small_config's default 2) means one
        # fresh story already meets keep_at_least, so drop_recently_published
        # has no reason to restore the repeat to avoid starving the edition —
        # otherwise this test would pass for the wrong reason.
        test_config = small_config(config)
        test_config.ai_daily.story_count = 1
        test_config.ai_daily.min_story_count = 1

        parse = feeds_for({
            "ars": [
                entry("A repeat AI story", "https://ars.example/repeat"),
                entry("A fresh AI story", "https://ars.example/fresh"),
            ],
            "wired": [],
        })
        editions = [{
            "date": NOW.strftime("%Y-%m-%d"),
            "generated_at": NOW.isoformat(),
            "stories": [{"title": "A repeat AI story", "url": "https://ars.example/repeat"}],
        }]
        briefer = FakeBriefer(BriefVerdict(picks=[]))
        run(test_config, briefer, parse=parse, editions=editions)
        assert "A repeat AI story" not in briefer.seen_candidates
        assert "A fresh AI story" in briefer.seen_candidates


class TestAiEditionStorage:
    def brief(self, n=1):
        from lastweekintech.domain import AiBrief

        return AiBrief(
            title=f"Story {n}",
            articles=[make_article(title=f"Story {n}", url=f"https://example.com/{n}")],
            theme="Capabilities",
            what_happened="It happened.",
            why_it_matters="It matters.",
            watch_next="Watch this.",
            score=10 - n,
        )

    def test_build_ai_edition_shapes_the_payload(self):
        edition = ai_daily.build_ai_edition([self.brief(1), self.brief(2)], date="2026-09-15", now=NOW)
        assert edition["date"] == "2026-09-15"
        assert len(edition["stories"]) == 2
        assert edition["stories"][0]["rank"] == 1
        assert edition["stories"][0]["theme"] == "Capabilities"
        assert edition["stories"][0]["what_happened"] == "It happened."
        assert edition["stories"][0]["why_it_matters"] == "It matters."
        assert edition["stories"][0]["watch_next"] == "Watch this."
        # A flattened field for the syndication helpers this reuses.
        assert edition["stories"][0]["summary"] == "It happened."

    def test_save_and_list_round_trip(self, tmp_path):
        edition = ai_daily.build_ai_edition([self.brief()], date="2026-09-15", now=NOW)
        ai_daily.save_ai_edition(edition, tmp_path)
        assert (tmp_path / "ai" / "latest.json").exists()
        assert (tmp_path / "ai" / "archive" / "2026-09-15.json").exists()

        editions = ai_daily.list_ai_editions(tmp_path)
        assert len(editions) == 1
        assert editions[0]["date"] == "2026-09-15"

    def test_list_editions_is_empty_when_nothing_published_yet(self, tmp_path):
        assert ai_daily.list_ai_editions(tmp_path) == []

    def test_list_editions_sorts_newest_first(self, tmp_path):
        for date in ["2026-09-13", "2026-09-15", "2026-09-14"]:
            ai_daily.save_ai_edition(
                ai_daily.build_ai_edition([self.brief()], date=date, now=NOW), tmp_path
            )
        editions = ai_daily.list_ai_editions(tmp_path)
        assert [e["date"] for e in editions] == ["2026-09-15", "2026-09-14", "2026-09-13"]
```

Note: `test_build_digest.py` already defines `FakeEntry`, `entry`, `feeds_for`; import them rather than redefining. Check the exact import path — `from test_fetch import FakeEntry` is used inside `test_build_digest.py` itself (per Task 4's context reading), so `FakeEntry` actually lives in `test_fetch.py`. Fix the import at the top of the new test file:

```python
from test_fetch import FakeEntry
```

and drop `FakeEntry` from the `test_build_digest` import line, leaving:

```python
from test_build_digest import entry, feeds_for
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_ai_daily.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lastweekintech.ai_daily'`

- [ ] **Step 4: Write `ai_daily.py` (orchestration and storage parts)**

Create `src/lastweekintech/ai_daily.py`:

```python
"""Orchestration for AI Daily: a second, independent pipeline.

Reuses the weekly digest's fetch, dedupe, cluster, score and extract stages
unmodified — each takes a Config rather than reading module state, so a
derived Config (a 24-hour window, daily-tuned HN thresholds) flows through
them without any changes to those functions. It diverges after extraction:
no mechanical category filter narrows the candidate pool (a chip
export-control story would be tagged Hardware or Policy, not AI, so
relevance is judged by the briefing model itself against a broad pool), and
one model call (``briefer.Briefer``) both selects the day's stories and
writes the analysis for each, rather than an editor-picks-then-summarizer-
writes split.
"""

import dataclasses
import json
import logging
import shutil
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from lastweekintech import hn, syndication
from lastweekintech.briefer import BriefPick, Briefer, BriefVerdict
from lastweekintech.config import Config
from lastweekintech.domain import AiBrief, AiDigest, Article, Story
from lastweekintech.pipeline import (
    ARCHIVE_DIRNAME,
    PACKAGE_DIR,
    cluster_articles,
    dedupe_articles,
    drop_recently_published,
    extract_content,
    fetch_articles,
    score_stories,
)

AI_DAILY_SUBDIR = "ai"
DEFAULT_SITE_URL = "https://lastweekin.tech"


def build_ai_daily(
    config: Config,
    briefer: Briefer,
    now: datetime | None = None,
    parse: Callable[[str], Any] | None = None,
    download: Callable[[str], str] | None = None,
    hn_fetch: hn.JsonFetcher | None = None,
    delay: float = 0.5,
    editions: list[dict[str, Any]] | None = None,
) -> AiDigest:
    """Run the daily AI curation stages and return the day's briefing.

    ``editions`` is the AI Daily archive (not the weekly one), used both to
    keep already-published stories out of the candidate pool and to tell the
    briefer what was recently covered, so it can distinguish a rehash from a
    material update.
    """
    now = now or datetime.now(UTC)
    daily_config = dataclasses.replace(
        config,
        window_days=config.ai_daily.window_days,
        hn=config.ai_daily.hn,
        weights=config.ai_daily.weights,
    )

    feed_articles = fetch_articles(daily_config, now=now, parse=parse)
    hn_articles = hn.fetch_hn_articles(daily_config, now=now, fetch=hn_fetch)
    articles = dedupe_articles(feed_articles + hn_articles)
    hn.merge_hn_points(articles, hn_articles)

    stories = cluster_articles(articles)
    stories = score_stories(stories, daily_config, now=now)

    ranked = drop_recently_published(
        stories,
        _for_shared_helpers(editions or []),
        now=now,
        lookback=timedelta(days=config.ai_daily.repeat_lookback_days),
        keep_at_least=config.ai_daily.story_count,
    )

    candidates = ranked[: config.ai_daily.candidate_pool]
    candidate_articles = [a for s in candidates for a in s.articles]
    extract_content(candidate_articles, download=download, delay=delay)

    recent_titles = _recent_titles(editions or [], now, config.ai_daily.repeat_lookback_days)
    verdict = briefer.brief(
        candidates,
        count=config.ai_daily.story_count,
        min_count=config.ai_daily.min_story_count,
        max_per_source=config.ai_daily.max_per_source,
        recent_titles=recent_titles,
    )
    if verdict is None:
        logging.error("No AI Daily briefing model produced a usable verdict; nothing to publish.")
        return AiDigest(briefs=[], intro=None)

    briefs = select_ai_daily_edition(candidates, verdict)
    return AiDigest(briefs=briefs, intro=verdict.intro or None)


def select_ai_daily_edition(candidates: list[Story], verdict: BriefVerdict) -> list[AiBrief]:
    """Turn the briefer's verdict into the day's briefs, in its chosen order.

    Unlike the weekly editor, there is no mechanical fallback ranking to
    reorder into: ``Briefer.brief()`` already rejects any verdict that breaks
    the source cap or the pick-count floor/ceiling before this is ever
    called, so every pick here is already valid to publish as-is.
    """
    return [_to_ai_brief(candidates[pick.n - 1], pick) for pick in verdict.picks]


def _to_ai_brief(story: Story, pick: BriefPick) -> AiBrief:
    return AiBrief(
        title=story.title,
        articles=story.articles,
        theme=pick.theme,
        what_happened=pick.what_happened,
        why_it_matters=pick.why_it_matters,
        watch_next=pick.watch_next,
        score=story.score,
    )


def build_ai_edition(
    briefs: list[AiBrief],
    date: str,
    now: datetime | None = None,
    intro: str | None = None,
) -> dict[str, Any]:
    """Assemble the published JSON payload for one day's AI briefing."""
    now = now or datetime.now(UTC)
    entries = []

    for rank, brief in enumerate(briefs, start=1):
        main_article = _representative_article(brief.articles)
        sources = sorted({a.source for a in brief.articles})
        entries.append({
            "rank": rank,
            "title": brief.title,
            "source": main_article.source if main_article else "",
            "sources": sources,
            "source_count": len(sources),
            "url": main_article.url if main_article else "",
            "theme": brief.theme,
            "hn_points": main_article.hn_points if main_article else None,
            "score": round(brief.score, 3),
            "what_happened": brief.what_happened,
            "why_it_matters": brief.why_it_matters,
            "watch_next": brief.watch_next,
            # A flattened summary, for the syndication helpers shared with
            # the weekly digest (_stories_html renders a single paragraph).
            "summary": brief.what_happened,
        })

    return {
        "date": date,
        "generated_at": now.isoformat(),
        "intro": intro or "",
        "stories": entries,
    }


def _representative_article(articles: list[Article]) -> Article | None:
    if not articles:
        return None
    return max(
        articles,
        key=lambda a: (not a.aggregator, a.hn_points or 0, len(a.content or "")),
    )


def save_ai_edition(edition: dict[str, Any], data_dir: Path) -> tuple[Path, Path]:
    """Write the edition to ``data_dir/ai/latest.json`` and its archive copy."""
    ai_dir = data_dir / AI_DAILY_SUBDIR
    archive_dir = ai_dir / ARCHIVE_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)

    latest_path = ai_dir / "latest.json"
    archive_path = archive_dir / f"{edition['date']}.json"
    payload = json.dumps(edition, indent=2, ensure_ascii=False) + "\n"

    latest_path.write_text(payload, encoding="utf-8")
    archive_path.write_text(payload, encoding="utf-8")
    logging.info(f"Saved AI Daily edition to {latest_path} and {archive_path}")
    return latest_path, archive_path


def list_ai_editions(data_dir: Path) -> list[dict[str, Any]]:
    """Return every archived AI Daily edition, newest first."""
    archive_dir = data_dir / AI_DAILY_SUBDIR / ARCHIVE_DIRNAME
    if not archive_dir.is_dir():
        return []

    editions = []
    for path in sorted(archive_dir.glob("*.json")):
        try:
            edition = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Skipping unreadable AI Daily archive entry {path}: {e}")
            continue
        if isinstance(edition, dict) and edition.get("date"):
            editions.append(edition)

    return sorted(editions, key=lambda e: e["date"], reverse=True)


def _recent_titles(editions: list[dict[str, Any]], now: datetime, lookback_days: int) -> list[str]:
    """Titles from the last ``lookback_days`` of the AI Daily archive.

    Purely for the briefing prompt's "avoid repeating" context — the
    mechanical dedupe against these same editions runs separately, through
    ``drop_recently_published`` (see ``_for_shared_helpers``).
    """
    if lookback_days < 1:
        return []
    cutoff = now.date() - timedelta(days=lookback_days)
    titles: list[str] = []
    for edition in editions:
        if not isinstance(edition, dict):
            continue
        published = _parse_date(edition.get("date"))
        if published is None or published < cutoff:
            continue
        for entry in edition.get("stories") or []:
            if isinstance(entry, dict) and isinstance(entry.get("title"), str):
                titles.append(entry["title"])
    return titles


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _for_shared_helpers(editions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Alias each edition's ``date`` as ``week`` for functions shared with the
    weekly pipeline (``drop_recently_published``), which key off ``week`` —
    reusing the field name avoids threading a new parameter through code the
    weekly digest also depends on, and a ``YYYY-MM-DD`` value round-trips
    identically whichever pipeline produced it.
    """
    return [{**e, "week": e.get("date")} for e in editions if isinstance(e, dict)]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_ai_daily.py -v`
Expected: PASS, all tests.

- [ ] **Step 6: Run the full suite, lint, and commit**

Run: `uv run pytest -q && uv run ruff check src/lastweekintech/ai_daily.py src/lastweekintech/domain.py tests/test_ai_daily.py`
Expected: PASS, no lint errors.

```bash
git add src/lastweekintech/domain.py src/lastweekintech/ai_daily.py tests/test_ai_daily.py
git commit -m "feat: add ai_daily orchestration, domain types and storage"
```

---

### Task 6: Site rendering — template and `generate_ai_site`

**Files:**
- Create: `src/lastweekintech/templates/ai_daily.html.jinja`
- Modify: `src/lastweekintech/templates/edition.html.jinja`
- Modify: `src/lastweekintech/ai_daily.py`
- Test: `tests/test_ai_daily.py` (append)

**Interfaces:**
- Consumes: `write_feed`/`write_sitemap` (Task 3, the latter unmodified), `_for_shared_helpers` (Task 5), `PACKAGE_DIR` (from `pipeline.py`).
- Produces: `generate_ai_site(editions, output_dir, template_dir=None, static_dir=None, site_url=DEFAULT_SITE_URL) -> list[Path]`. Task 7's CLI calls this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ai_daily.py`:

```python
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "src" / "lastweekintech" / "templates"
STATIC_DIR = Path(__file__).resolve().parents[1] / "src" / "lastweekintech" / "static"


def ai_edition(date="2026-09-15", title="OpenAI ships a new model", theme="Capabilities"):
    from lastweekintech.domain import AiBrief

    brief = AiBrief(
        title=title,
        articles=[make_article(title=title, source="Ars Technica")],
        theme=theme,
        what_happened="A factual account of what happened.",
        why_it_matters="A paragraph on why this matters and what it implies.",
        watch_next="What to watch for next.",
    )
    return ai_daily.build_ai_edition([brief], date=date, now=NOW)


def generate(tmp_path, editions):
    return ai_daily.generate_ai_site(
        editions, output_dir=tmp_path, template_dir=TEMPLATE_DIR, static_dir=STATIC_DIR
    )


class TestGenerateAiSite:
    def test_writes_the_latest_edition_under_ai(self, tmp_path):
        generate(tmp_path, [ai_edition()])
        page = (tmp_path / "ai" / "index.html").read_text()
        assert "OpenAI ships a new model" in page
        assert "A paragraph on why this matters" in page
        assert "What to watch for next" in page

    def test_writes_a_page_per_archived_edition(self, tmp_path):
        generate(tmp_path, [ai_edition(date="2026-09-15"), ai_edition(date="2026-09-14")])
        assert (tmp_path / "ai" / "archive" / "2026-09-14.html").exists()
        assert (tmp_path / "ai" / "archive" / "2026-09-15.html").exists()

    def test_writes_its_own_feed_and_sitemap(self, tmp_path):
        generate(tmp_path, [ai_edition()])
        assert (tmp_path / "ai" / "feed.xml").exists()
        assert (tmp_path / "ai" / "sitemap.xml").exists()
        assert "AI Daily" in (tmp_path / "ai" / "feed.xml").read_text()

    def test_copies_static_assets_under_ai_too(self, tmp_path):
        generate(tmp_path, [ai_edition()])
        assert (tmp_path / "ai" / "style.css").exists()

    def test_archive_page_links_back_to_the_ai_index_and_weekly_digest(self, tmp_path):
        generate(tmp_path, [ai_edition(date="2026-09-15"), ai_edition(date="2026-09-14")])
        page = (tmp_path / "ai" / "archive" / "2026-09-14.html").read_text()
        assert 'href="../index.html"' in page
        assert 'href="../../index.html"' in page

    def test_no_editions_writes_nothing(self, tmp_path):
        assert ai_daily.generate_ai_site([], output_dir=tmp_path) == []
        assert not (tmp_path / "ai").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ai_daily.py -k TestGenerateAiSite -v`
Expected: FAIL — `AttributeError: module 'lastweekintech.ai_daily' has no attribute 'generate_ai_site'`

- [ ] **Step 3: Create the template**

Create `src/lastweekintech/templates/ai_daily.html.jinja`:

```jinja
<!DOCTYPE html>
{#
  Optional template variables:
    site_url  absolute origin of this section, e.g. "https://lastweekin.tech/ai".
              When absent, absolute-URL metadata (canonical, og:image) is omitted
              rather than emitted as a broken relative URL.
#}
{% set site = (site_url | default('', true) | string).rstrip('/') %}
{% set page_path = '' if is_latest else 'archive/' ~ edition.date ~ '.html' %}
{% set weekly_root = root ~ '../' %}
{% macro pretty_date(value, with_year=true) -%}
  {%- set months = {
    '01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
    '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'
  } -%}
  {%- set value = value | string -%}
  {%- if value | length == 10 and months.get(value[5:7]) -%}
    {{- months[value[5:7]] }} {{ value[8:10] | int }}{{ ', ' ~ value[:4] if with_year else '' -}}
  {%- else -%}
    {{- value -}}
  {%- endif -%}
{%- endmacro %}
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{{ page_title }}</title>
    <meta name="description" content="{{ description }}" />
    <meta name="color-scheme" content="light dark" />
    <meta property="og:site_name" content="LastWeekIn.Tech — AI Daily" />
    <meta property="og:title" content="{{ page_title }}" />
    <meta property="og:description" content="{{ description }}" />
    <meta property="og:type" content="website" />
    <meta name="twitter:card" content="summary_large_image" />
    {% if site %}
    <link rel="canonical" href="{{ site }}/{{ page_path }}" />
    <meta property="og:url" content="{{ site }}/{{ page_path }}" />
    <meta property="og:image" content="{{ site }}/og.png" />
    <meta property="og:image:width" content="1200" />
    <meta property="og:image:height" content="630" />
    <meta
      property="og:image:alt"
      content="LastWeekIn.Tech AI Daily — the AI stories that mattered today"
    />
    <meta name="twitter:image" content="{{ site }}/og.png" />
    {% endif %}
    <link
      rel="alternate"
      type="application/atom+xml"
      title="LastWeekIn.Tech AI Daily"
      href="{{ root }}feed.xml"
    />
    <link rel="stylesheet" href="{{ root }}style.css" />
    <link
      rel="icon"
      href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22><text y=%2214%22 font-size=%2214%22>7</text></svg>"
    />
  </head>
  <body>
    <a class="skip-link" href="#stories">Skip to the stories</a>

    <header class="site-header">
      <p class="masthead-note">Published daily &middot; AI only &middot; No trackers, no ads</p>
      <h1 class="site-title">
        <a class="wordmark" href="{{ root }}index.html"
          >LastWeekIn<span class="dot">.</span>Tech <span class="dot">AI Daily</span></a
        >
      </h1>
      <div class="dateline" role="presentation">
        <span class="dateline-item">The {{ edition.stories | length }} AI stories that mattered</span>
        <span class="dateline-item"
          >Briefing for
          <time datetime="{{ edition.date }}">{{ pretty_date(edition.date) }}</time></span
        >
      </div>
      {% if not is_latest %}
      <p class="archive-note">
        An archived briefing. <a href="{{ root }}index.html">See today&rsquo;s briefing.</a>
      </p>
      {% endif %}
    </header>

    <main>
      {% if edition.intro %}
      <p class="standfirst">{{ edition.intro }}</p>
      {% endif %}

      <ol class="story-list" id="stories">
        {% for story in edition.stories %}
        {% set theme = story.theme | default('', true) | string %}
        {% set theme_slug = theme | lower | replace(' & ', '-') | replace(' ', '-') %}
        <li class="story-card{{ ' lead-story' if loop.first else '' }}">
          <p class="story-rank" aria-hidden="true">{{ story.rank }}</p>
          <div class="story-content">
            <p class="story-meta">
              <span class="visually-hidden">Number {{ story.rank }}. </span>
              {% if theme %}
              <span class="story-category category-{{ theme_slug }}">{{ theme }}</span>
              {% endif %}
              <span class="story-source">{{ story.source }}</span>
            </p>
            <h2 class="story-title">
              {% if story.url %}
              <a href="{{ story.url }}" target="_blank" rel="noopener noreferrer"
                >{{ story.title }}</a
              >
              {% else %}
              {{ story.title }}
              {% endif %}
            </h2>
            {% if story.what_happened %}
            <p class="story-summary">{{ story.what_happened }}</p>
            {% endif %}
            {% if story.why_it_matters %}
            <p class="story-case">
              <span class="story-case-label">Why it matters</span> {{ story.why_it_matters }}
            </p>
            {% endif %}
            {% if story.watch_next %}
            <p class="story-case">
              <span class="story-case-label">What to watch</span> {{ story.watch_next }}
            </p>
            {% endif %}
          </div>
        </li>
        {% endfor %}
      </ol>

      <section class="subscribe" aria-labelledby="subscribe-heading">
        <h2 id="subscribe-heading">Get it every day</h2>
        <p>
          A short AI briefing, every day.
          <a href="{{ root }}feed.xml">Subscribe with RSS</a> — no email address, no tracking.
        </p>
      </section>

      {% if past_editions %}
      <nav class="archive" aria-labelledby="archive-heading">
        <h2 id="archive-heading">Past briefings</h2>
        {% set years = [] %}
        {% for past in past_editions %}
        {% if (past.date | string)[:4] not in years %}
        {% set _ = years.append((past.date | string)[:4]) %}
        {% endif %}
        {% endfor %}
        {% for year in years %}
        {% set in_year = [] %}
        {% for past in past_editions %}
        {% if (past.date | string)[:4] == year %}
        {% set _ = in_year.append(past) %}
        {% endif %}
        {% endfor %}
        <details class="archive-year"{{ ' open' if loop.first else '' }}>
          <summary>
            <span class="archive-year-label">{{ year }}</span>
            <span class="archive-count"
              >{{ in_year | length }} briefing{{ '' if in_year | length == 1 else 's' }}</span
            >
          </summary>
          <ul>
            {% for past in in_year %}
            <li>
              <a href="{{ root }}archive/{{ past.date }}.html"
                >{{ pretty_date(past.date, with_year=false) }}</a
              >
            </li>
            {% endfor %}
          </ul>
        </details>
        {% endfor %}
      </nav>
      {% endif %}
    </main>

    <footer>
      <p>
        Curated automatically from {{ edition.stories | map(attribute='source') | unique | list |
        length }} sources. No trackers, no ads.
      </p>
      <p class="generated">Generated {{ edition.generated_at[:10] }}.</p>
      <p class="ai-daily-link">
        <a href="{{ weekly_root }}index.html">Also: the weekly Top-7 digest</a>
      </p>
    </footer>
  </body>
</html>
```

- [ ] **Step 4: Add the small link from the weekly template**

In `src/lastweekintech/templates/edition.html.jinja`, in the `<footer>` block, add one line after `<p class="generated">`:

```html
    <footer>
      <p>
        Curated automatically from {{ edition.stories | map(attribute='source') | unique | list |
        length }} sources. No trackers, no ads.
      </p>
      <p class="generated">Generated {{ edition.generated_at[:10] }}.</p>
      <p class="ai-daily-link"><a href="{{ root }}ai/">Also: AI Daily, a shorter briefing every day</a></p>
    </footer>
```

- [ ] **Step 5: Add `generate_ai_site` and `_render_ai_page` to `ai_daily.py`**

In `src/lastweekintech/ai_daily.py`, add `import shutil` if not already present (it is, from Step 4 of Task 5), and append at the end of the file:

```python
def generate_ai_site(
    editions: list[dict[str, Any]],
    output_dir: Path,
    template_dir: Path | None = None,
    static_dir: Path | None = None,
    site_url: str = DEFAULT_SITE_URL,
) -> list[Path]:
    """Render the AI Daily briefing as a static site under ``output_dir/ai``.

    Mirrors ``pipeline.generate_site``'s structure, with its own template and
    its own feed/sitemap at the ``/ai`` sub-path — see ``_for_shared_helpers``
    for how ``syndication.write_feed``/``write_sitemap`` are reused unmodified
    despite AI Daily's edition dicts using ``date`` instead of ``week``.
    """
    if not editions:
        logging.warning("No AI Daily editions to render; skipping site generation.")
        return []

    template_dir = template_dir or PACKAGE_DIR / "templates"
    static_dir = static_dir or PACKAGE_DIR / "static"
    ai_output_dir = output_dir / AI_DAILY_SUBDIR
    editions = sorted(editions, key=lambda e: e["date"], reverse=True)
    ai_site_url = f"{site_url.rstrip('/')}/{AI_DAILY_SUBDIR}"

    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    template = env.get_template("ai_daily.html.jinja")

    latest, past = editions[0], editions[1:]
    written = [
        _render_ai_page(
            template,
            ai_output_dir / "index.html",
            latest,
            past,
            root="",
            is_latest=True,
            site_url=ai_site_url,
        )
    ]

    archive_dir = ai_output_dir / ARCHIVE_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)
    for edition in editions:
        written.append(
            _render_ai_page(
                template,
                archive_dir / f"{edition['date']}.html",
                edition,
                [e for e in editions if e["date"] != edition["date"]],
                root="../",
                is_latest=False,
                site_url=ai_site_url,
            )
        )

    shared_editions = _for_shared_helpers(editions)
    written.append(
        syndication.write_feed(
            shared_editions,
            ai_output_dir,
            ai_site_url,
            title="LastWeekIn.Tech — AI Daily",
            subtitle="The 4-5 AI developments that mattered today.",
            entry_title=lambda date: f"AI Daily — {date}",
        )
    )
    written.append(syndication.write_sitemap(shared_editions, ai_output_dir, ai_site_url))

    for asset in sorted(static_dir.glob("*")):
        if asset.is_file():
            shutil.copy(asset, ai_output_dir / asset.name)

    logging.info(f"Generated {len(written)} AI Daily pages in {ai_output_dir}")
    return written


def _render_ai_page(
    template: Any,
    path: Path,
    edition: dict[str, Any],
    past_editions: list[dict[str, Any]],
    root: str,
    is_latest: bool,
    site_url: str,
) -> Path:
    edition_date = edition["date"]
    page_title = (
        "LastWeekIn.Tech — AI Daily"
        if is_latest
        else f"LastWeekIn.Tech — AI Daily, {edition_date}"
    )
    html = template.render(
        edition=edition,
        past_editions=past_editions,
        root=root,
        is_latest=is_latest,
        site_url=site_url,
        page_title=page_title,
        description=(
            f"The {len(edition['stories'])} AI developments that mattered on {edition_date}, "
            "with analysis and what to watch next."
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_ai_daily.py -v`
Expected: PASS, all tests including every `TestGenerateAiSite` case.

- [ ] **Step 7: Run the full suite (including the existing weekly site tests, to confirm the footer addition didn't break anything)**

Run: `uv run pytest tests/test_site.py tests/test_ai_daily.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/lastweekintech/templates/ai_daily.html.jinja src/lastweekintech/templates/edition.html.jinja src/lastweekintech/ai_daily.py tests/test_ai_daily.py
git commit -m "feat: render the AI Daily site, with its own feed and sitemap"
```

---

### Task 7: CLI — `run` and `ai-daily` subcommands

**Files:**
- Modify: `src/lastweekintech/main.py`
- Modify: `tests/test_cli.py`
- Modify: `.github/workflows/main.yml`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `ai_daily.build_ai_daily`, `ai_daily.build_ai_edition`, `ai_daily.save_ai_edition`, `ai_daily.list_ai_editions`, `ai_daily.generate_ai_site`, `ai_daily.AI_DAILY_SUBDIR` (Tasks 5-6); `Briefer` (Task 4).
- Produces: two Typer commands, `run` and `ai-daily`, callable as `uv run lastweekintech run [...]` and `uv run lastweekintech ai-daily [...]`.

- [ ] **Step 1: Update `test_cli.py`'s shared `invoke()` helper and existing calls**

In `tests/test_cli.py`, Typer now has two registered commands, so every invocation must name one. Change `invoke()`:

```python
def invoke(tmp_path, *args):
    return runner.invoke(
        main.app,
        ["run", "--data-dir", str(tmp_path / "data"), "--site-dir", str(tmp_path), *args],
    )
```

No other line in the file changes — every existing test calls `invoke(tmp_path, ...)`, which now prepends `"run"` automatically.

- [ ] **Step 2: Run the existing CLI tests to verify they still fail correctly (proving the subcommand requirement is real)**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — every test errors, because `main.app` still only has the one `run` command registered under its default (unnamed) behavior; `runner.invoke(main.app, ["run", ...])` currently means "pass the literal argument `run`" to the single implicit command, which Typer will reject as an unexpected argument. This confirms the test file is now written for the two-command shape the next step introduces.

- [ ] **Step 3: Add the `ai-daily` command and make `run`'s name explicit**

In `src/lastweekintech/main.py`, update the imports:

```python
from lastweekintech import ai_daily, metrics, pipeline
from lastweekintech.briefer import Briefer
from lastweekintech.config import ConfigError, get_config
from lastweekintech.editor import Editor
from lastweekintech.summarizer import Summarizer
from lastweekintech.validation import DigestValidationError, assert_publishable
```

Change the decorator on the existing command from `@app.command()` to `@app.command("run")` (this makes the subcommand name explicit rather than relying on the function name, and — combined with a second command existing — this is what requires callers to type `run`):

```python
@app.command("run")
def run(
    ...
```

Leave the rest of `run`'s body untouched. Then add the new command after it, before the final `if __name__ == "__main__":` block:

```python
@app.command("ai-daily")
def ai_daily_command(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Where the AI Daily data lives."),
    ] = Path("data"),
    site_dir: Annotated[
        Path,
        typer.Option("--site-dir", "-s", help="Where the static site is written."),
    ] = Path("public"),
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config.yaml."),
    ] = None,
    date: Annotated[
        str | None,
        typer.Option("--date", help="Briefing date (YYYY-MM-DD). Defaults to today, UTC."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild and overwrite an already-published day."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the briefing instead of writing any files."),
    ] = False,
):
    """Run the AI Daily briefing pipeline."""
    now = datetime.now(UTC)
    date = date or now.strftime("%Y-%m-%d")

    already_published = data_dir / ai_daily.AI_DAILY_SUBDIR / "archive" / f"{date}.json"
    if already_published.exists() and not force and not dry_run:
        typer.secho(
            f"The {date} AI Daily briefing is already published ({already_published}); "
            "use --force to rebuild it.",
            fg=typer.colors.YELLOW,
        )
        return

    try:
        config = get_config(config_path)
    except ConfigError as e:
        typer.secho(f"Configuration error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e

    if not config.ai_daily.enabled:
        typer.secho("AI Daily is disabled in config.yaml.", fg=typer.colors.YELLOW)
        return

    try:
        briefer = Briefer(config.ai_daily.briefing)
    except ValueError as e:
        typer.secho(f"Configuration error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e

    digest = ai_daily.build_ai_daily(
        config,
        briefer,
        now=now,
        editions=ai_daily.list_ai_editions(data_dir),
    )

    if not digest.briefs:
        typer.secho(
            "No AI Daily briefing model produced a usable verdict; nothing published today.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    edition = ai_daily.build_ai_edition(digest.briefs, date=date, now=now, intro=digest.intro)

    if dry_run:
        typer.echo(json.dumps(edition, indent=2, ensure_ascii=False))
        return

    ai_daily.save_ai_edition(edition, data_dir)
    ai_daily.generate_ai_site(
        ai_daily.list_ai_editions(data_dir),
        output_dir=site_dir,
        site_url=config.site_url,
    )
    typer.secho(f"Published the AI Daily briefing for {date}.", fg=typer.colors.GREEN)
```

- [ ] **Step 4: Run the existing CLI tests to verify they now pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS, every existing test (now routed through the explicit `"run"` subcommand).

- [ ] **Step 5: Write the failing tests for the new `ai-daily` command**

Append to `tests/test_cli.py`:

```python
@pytest.fixture
def stub_ai_daily(monkeypatch, config):
    """Run the ai-daily CLI against a canned digest instead of the network."""

    def briefs(count=5):
        from lastweekintech.domain import AiBrief

        return [
            AiBrief(
                title=f"AI story {i}",
                articles=[make_article(url=f"https://example.com/ai/{i}")],
                theme="Capabilities",
                what_happened="It happened.",
                why_it_matters="It matters.",
                watch_next="Watch this.",
                score=100 - i,
            )
            for i in range(count)
        ]

    state = {"briefs": briefs()}
    monkeypatch.setattr(main, "get_config", lambda path=None: config)
    monkeypatch.setattr(main, "Briefer", lambda settings: object())
    monkeypatch.setattr(
        main.ai_daily,
        "build_ai_daily",
        lambda *a, **k: main.ai_daily.AiDigest(briefs=state["briefs"]),
    )
    state["make"] = briefs
    return state


def invoke_ai_daily(tmp_path, *args):
    return runner.invoke(
        main.app,
        ["ai-daily", "--data-dir", str(tmp_path / "data"), "--site-dir", str(tmp_path), *args],
    )


class TestAiDailyCommand:
    def test_publishes_a_good_briefing(self, tmp_path, stub_ai_daily):
        result = invoke_ai_daily(tmp_path, "--date", "2026-09-15")
        assert result.exit_code == 0
        assert (tmp_path / "ai" / "index.html").exists()
        assert (tmp_path / "data" / "ai" / "archive" / "2026-09-15.json").exists()

    def test_fails_when_no_model_produced_a_verdict(self, tmp_path, stub_ai_daily):
        stub_ai_daily["briefs"] = []
        result = invoke_ai_daily(tmp_path, "--date", "2026-09-15")
        assert result.exit_code != 0
        assert not (tmp_path / "ai" / "index.html").exists()

    def test_dry_run_writes_nothing(self, tmp_path, stub_ai_daily):
        result = invoke_ai_daily(tmp_path, "--date", "2026-09-15", "--dry-run")
        assert result.exit_code == 0
        assert not (tmp_path / "ai").exists()
        assert not (tmp_path / "data").exists()
        assert "AI story 0" in result.stdout

    def test_skips_a_day_that_is_already_published(self, tmp_path, stub_ai_daily):
        archive = tmp_path / "data" / "ai" / "archive"
        archive.mkdir(parents=True)
        (archive / "2026-09-15.json").write_text('{"date": "2026-09-15", "stories": []}')

        result = invoke_ai_daily(tmp_path, "--date", "2026-09-15")
        assert result.exit_code == 0
        assert "already published" in result.output
        assert not (tmp_path / "ai" / "index.html").exists()

    def test_force_rebuilds_a_published_day(self, tmp_path, stub_ai_daily):
        archive = tmp_path / "data" / "ai" / "archive"
        archive.mkdir(parents=True)
        (archive / "2026-09-15.json").write_text('{"date": "2026-09-15", "stories": []}')

        result = invoke_ai_daily(tmp_path, "--date", "2026-09-15", "--force")
        assert result.exit_code == 0
        assert (tmp_path / "ai" / "index.html").exists()

    def test_disabled_in_config_does_nothing(self, tmp_path, stub_ai_daily, config):
        config.ai_daily.enabled = False
        result = invoke_ai_daily(tmp_path, "--date", "2026-09-15")
        assert result.exit_code == 0
        assert "disabled" in result.output
        assert not (tmp_path / "ai").exists()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS, all tests including the new `TestAiDailyCommand` cases.

- [ ] **Step 7: Update `main.yml`'s pipeline step**

In `.github/workflows/main.yml`, change:

```yaml
          uv run lastweekintech $SKIP_GATE_FLAG 2>&1 | tee "$log"
```

to:

```yaml
          uv run lastweekintech run $SKIP_GATE_FLAG 2>&1 | tee "$log"
```

- [ ] **Step 8: Update `CLAUDE.md`'s Commands section**

In `CLAUDE.md`, change:

```
uv run lastweekintech                  # full run → data/ + site
uv run lastweekintech --dry-run        # print the edition, write nothing
```

to:

```
uv run lastweekintech run              # full weekly run → data/ + site
uv run lastweekintech run --dry-run    # print the edition, write nothing
uv run lastweekintech ai-daily         # full AI Daily run → data/ai/ + site/ai
uv run lastweekintech ai-daily --dry-run  # print the briefing, write nothing
```

- [ ] **Step 9: Run the full suite, lint, and commit**

Run: `uv run pytest -q && uv run ruff check src/lastweekintech/main.py tests/test_cli.py`
Expected: PASS, no lint errors.

```bash
git add src/lastweekintech/main.py tests/test_cli.py .github/workflows/main.yml CLAUDE.md
git commit -m "feat: add the ai-daily CLI command; run is now an explicit subcommand"
```

---

### Task 8: `tools/check_models.py` — verify the briefing models too

**Files:**
- Modify: `tools/check_models.py`

**Interfaces:**
- Consumes: `config.ai_daily.briefing.model_name`, `config.ai_daily.briefing.fallback_models` (Task 1).

There is no existing test file for this script (it is a small ops tool, run manually and on its own schedule per `CLAUDE.md`); verify it by running it directly against the live config.

- [ ] **Step 1: Add the briefing models to the checked list**

In `tools/check_models.py`, change:

```python
    # Hugging Face models are routed elsewhere and are not in this catalogue.
    hosted = [summarizer.model_name, *summarizer.fallback_models]
    checked = [m for m in hosted if m not in set(summarizer.huggingface_models)]
```

to:

```python
    briefing = config.ai_daily.briefing
    # Hugging Face models are routed elsewhere and are not in this catalogue.
    hosted = [
        summarizer.model_name,
        *summarizer.fallback_models,
        briefing.model_name,
        *briefing.fallback_models,
    ]
    checked = [m for m in hosted if m not in set(summarizer.huggingface_models)]
```

- [ ] **Step 2: Run it against the live config**

Run: `uv run python tools/check_models.py`
Expected: exits 0, and the printed list now includes `anthropic/claude-sonnet-5` (the `ai_daily.briefing.model_name`) alongside the summarizer's models — it will already show `ok` for it since it is also the weekly `editor.model_name`, already live.

- [ ] **Step 3: Commit**

```bash
git add tools/check_models.py
git commit -m "feat: check_models.py also verifies the AI Daily briefing models"
```

---

### Task 9: GitHub Actions — the daily workflow

**Files:**
- Create: `.github/workflows/ai-daily.yml`
- Modify: `.github/workflows/deploy-pages.yml`

**Interfaces:** none (infrastructure only; no automated test — verify with `actionlint` if available, otherwise careful manual review against `main.yml`'s already-working shape).

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/ai-daily.yml`:

```yaml
name: AI Daily Briefing

on:
  schedule:
    - cron: "0 13 * * *" # Every day at 1:00 PM UTC — clear of the Monday 08:00 UTC weekly run.
  workflow_dispatch:

# Independent of the weekly digest's own concurrency group: the two never need to
# block each other, and their schedules never coincide.
permissions: {}

concurrency:
  group: ai-daily-digest
  cancel-in-progress: false

jobs:
  build-brief:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    permissions:
      contents: write

    outputs:
      failure_reason: ${{ steps.pipeline.outputs.failure_reason }}

    steps:
      - name: Checkout repository
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1

      - name: Install uv
        uses: astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d # v10.0.1
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv sync

      - name: Run pipeline
        id: pipeline
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: |
          log="$RUNNER_TEMP/ai-daily.log"
          set +e
          uv run lastweekintech ai-daily 2>&1 | tee "$log"
          status=${PIPESTATUS[0]}
          set -e
          if [ "$status" -ne 0 ]; then
            delimiter="LWIT_AI_DAILY_$(openssl rand -hex 16)"
            {
              printf 'failure_reason<<%s\n' "$delimiter"
              tail -n 30 "$log" | sed 's/\x1b\[[0-9;]*m//g'
              printf '\n%s\n' "$delimiter"
            } >> "$GITHUB_OUTPUT"
          fi
          exit "$status"

      - name: Commit and push changes
        run: |
          git config --global user.name 'github-actions[bot]'
          git config --global user.email 'github-actions[bot]@users.noreply.github.com'
          generated=(data public)
          present=()
          for path in "${generated[@]}"; do
            if [ -e "$path" ]; then present+=("$path"); fi
          done
          git add -- "${present[@]}"
          if git diff --staged --quiet; then
            echo "No changes to commit."
            exit 0
          fi
          git commit -m "docs: AI Daily briefing for $(date -u +'%Y-%m-%d')"
          git push

  notify-failure:
    needs: build-brief
    if: failure()
    runs-on: ubuntu-latest
    timeout-minutes: 5

    permissions:
      issues: write

    steps:
      - name: Open or update the AI Daily failure issue
        uses: actions/github-script@3a2844b7e9c422d3c10d287c895573f7108da1b3 # v9.0.0
        env:
          FAILURE_REASON: ${{ needs.build-brief.outputs.failure_reason }}
          RUN_URL: "${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"
        with:
          script: |
            const LABEL = 'ai-daily-failure';
            const { owner, repo } = context.repo;
            const runUrl = process.env.RUN_URL;
            const reason = (process.env.FAILURE_REASON || '').trim();
            const today = new Date().toISOString().slice(0, 10);

            const details = reason
              ? ['```', reason.slice(0, 5000), '```'].join('\n')
              : '_No pipeline output was captured — the job failed before or after the pipeline step._';

            const body = [
              `The AI Daily briefing run failed on ${today}, so nothing was published.`,
              'The site still serves the previous briefing.',
              '',
              `**Run:** ${runUrl}`,
              `**Trigger:** \`${context.eventName}\``,
              '',
              '<details><summary>Last lines of the pipeline log</summary>',
              '',
              details,
              '',
              '</details>',
            ].join('\n');

            const openIssues = await github.paginate(github.rest.issues.listForRepo, {
              owner,
              repo,
              state: 'open',
              labels: LABEL,
              per_page: 100,
            });
            const existing = openIssues.find((issue) => !issue.pull_request);

            if (existing) {
              await github.rest.issues.createComment({
                owner,
                repo,
                issue_number: existing.number,
                body,
              });
              core.notice(`Commented on the open failure issue #${existing.number}.`);
              return;
            }

            try {
              await github.rest.issues.createLabel({
                owner,
                repo,
                name: LABEL,
                color: 'b60205',
                description: 'The AI Daily briefing run failed',
              });
            } catch (error) {
              if (error.status !== 422) throw error;
            }

            const created = await github.rest.issues.create({
              owner,
              repo,
              title: `AI Daily briefing failed (${today})`,
              labels: [LABEL],
              body,
            });
            core.notice(`Opened failure issue #${created.data.number}.`);
```

Note: unlike `main.yml`, this workflow's `run pipeline` step does not set `HF_TOKEN` or `PERPLEXITY_API_KEY` — `Briefer` (Task 4) never routes through Hugging Face and never calls Perplexity, so neither secret is read by the `ai-daily` command. Only `OPENROUTER_API_KEY` is needed.

- [ ] **Step 2: Update `deploy-pages.yml` to also trigger on this workflow**

In `.github/workflows/deploy-pages.yml`, change:

```yaml
  workflow_run:
    workflows: ["Weekly Tech Digest"]
    types: [completed]
```

to:

```yaml
  workflow_run:
    workflows: ["Weekly Tech Digest", "AI Daily Briefing"]
    types: [completed]
```

- [ ] **Step 3: Validate the YAML**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ai-daily.yml'))" && uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/deploy-pages.yml'))"`
Expected: no output, no error (both files parse as valid YAML).

If `actionlint` is available locally (`which actionlint`), also run:

Run: `actionlint .github/workflows/ai-daily.yml .github/workflows/deploy-pages.yml`
Expected: no errors. If `actionlint` is not installed, skip this check — it is not part of this project's documented toolchain (see `CLAUDE.md`), so do not install it as part of this task.

- [ ] **Step 4: Run the full test suite one final time**

Run: `uv run pytest -q`
Expected: PASS, all tests (weekly + AI Daily).

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ai-daily.yml .github/workflows/deploy-pages.yml
git commit -m "feat: add the AI Daily briefing workflow and wire it into Pages deploy"
```

---

## After all tasks: final verification

- [ ] Run `uv run pytest -q` — expect all tests (weekly + AI Daily) passing, 0 failures.
- [ ] Run `uv run ruff format . && uv run ruff check --fix .` — expect no remaining issues.
- [ ] Run `uv run mypy src` — expect no new type errors introduced by this feature (pre-existing errors elsewhere, if any, are out of scope).
- [ ] Run `uv run lastweekintech ai-daily --dry-run` with a real `OPENROUTER_API_KEY` set (see `.env.example`) — confirm it prints a plausible 4-5 story JSON briefing, or fails closed with a clear message if the live models don't cooperate. This is the one step this plan cannot verify with fakes: it is the first time real article content and a real model see each other.
