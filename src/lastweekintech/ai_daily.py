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

from lastweekintech import discovery, hn, syndication
from lastweekintech.briefer import Briefer, BriefPick, BriefVerdict
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
    search: discovery.SearchFn | None = None,
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

    # A 14-feed, 24-hour pool recalls only what those feeds happened to
    # publish in that window — on a day when the best AI stories are niche
    # items no mainstream outlet picked up yet, that pool can end up wholly
    # Hacker-News-sourced. Perplexity's live web search corroborates (never
    # replaces) the mechanical ranking, exactly like the weekly digest's own
    # consensus stage, just with an AI-scoped prompt and a 24h window.
    consensus = discovery.fetch_consensus(
        daily_config, now=now, search=search, prompt_template=discovery.AI_DAILY_PROMPT_TEMPLATE
    )
    missed = discovery.apply_consensus_boost(stories, consensus, daily_config.weights.consensus)
    # A boost only helps a story already in the pool — most of what the wider
    # web corroborates is genuinely absent from a 14-feed, 24-hour fetch, not
    # just under-ranked in it. Ingest those as real candidates instead of
    # only recording that they exist: they still have to survive extraction
    # (a real body from the citation URL) and every validation Briefer
    # already applies, same as an organically-fetched story.
    new_from_consensus = discovery.stories_from_missed(missed, daily_config.weights.consensus)
    if consensus:
        stories = stories + new_from_consensus
        stories.sort(key=lambda s: s.score, reverse=True)
        logging.info(f"Added {len(new_from_consensus)} consensus-only candidates to the pool.")

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
            entry_kind="ai-daily",
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
        "LastWeekIn.Tech — AI Daily" if is_latest else f"LastWeekIn.Tech — AI Daily, {edition_date}"
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
