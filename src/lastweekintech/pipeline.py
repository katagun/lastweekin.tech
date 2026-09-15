"""
Core data pipeline for LastWeekIn.Tech.
"""

import calendar
import json
import logging
import math
import re
import shutil
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import feedparser
import requests
import trafilatura
from jinja2 import Environment, FileSystemLoader
from thefuzz import fuzz

from lastweekintech import discovery, hn, syndication
from lastweekintech import editor as editorial
from lastweekintech.config import Config
from lastweekintech.domain import Article, Digest, Story
from lastweekintech.metrics import RunMetrics, summarizer_model
from lastweekintech.summarizer import Summarizer
from lastweekintech.text import (
    AI_CATEGORY,
    CATEGORY_PRECEDENCE,
    GENERAL_CATEGORY,
    contains_category_terms,
    count_category_terms,
    normalize_url,
)

PACKAGE_DIR = Path(__file__).resolve().parent

ARCHIVE_DIRNAME = "archive"
DEFAULT_SITE_URL = "https://lastweekin.tech"

# A title mention is decisive; a body needs several mentions before a story is
# considered to be *about* a topic rather than merely mentioning it.
BODY_MENTION_THRESHOLD = 3
AI_BODY_MENTION_THRESHOLD = BODY_MENTION_THRESHOLD  # kept for existing callers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# A default user agent gets a weekly batch job 403'd by most outlets, and a run
# that is blocked extracts nothing at all.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 10

# Containers holding "more stories" lists. trafilatura picks the page's main
# content by ranking candidate containers, and on some layouts a recirculation
# block wins: The Register served the same 3,442-character "MOST POPULAR" list
# as the body of every article, so the pipeline would have filed and summarized
# a story from other stories' headlines. Pruning these first is not a per-site
# rule — the class names are the ordinary conventions for the widget.
_RECIRCULATION_CLASSES = ("related", "most-popular", "sidebar", "recirc")
_RECIRCULATION_XPATH = [
    "//*[contains(translate(@class,"
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),"
    f"'{name}')]"
    for name in _RECIRCULATION_CLASSES
]
# Some sites really do put the article inside a "related-content" wrapper, where
# pruning removes the story itself. The two cases separate on an absolute floor
# rather than on a ratio: when pruning is right it still leaves a whole article,
# and when it is wrong it leaves a scrap. A ratio looks equivalent on the sample
# that motivated this and is not — it silently keeps the recirculation block
# whenever that block is more than twice the length of the article.
_MIN_PRUNED_BODY = 500

# Comments are excluded: a long thread swamps the article, and both the category
# classifier and the summarizer read this text as if it were the reporting.
_EXTRACT_OPTIONS: dict[str, Any] = {
    "include_comments": False,
    "include_tables": True,
}


def fetch_articles(
    config: Config,
    now: datetime | None = None,
    parse: Callable[[str], Any] | None = None,
) -> list[Article]:
    """Fetch articles published inside the window from all configured RSS feeds."""
    parse = parse or feedparser.parse
    now = now or datetime.now(UTC)
    start_date = now - timedelta(days=config.window_days)

    articles = []
    for feed in config.feeds:
        logging.info(f"Fetching articles from {feed.name}...")
        try:
            parsed_feed = parse(feed.url)
        except Exception as e:  # noqa: BLE001 - one bad feed must not end the run
            logging.error(f"Failed to fetch or parse feed {feed.name}: {e}")
            continue

        for entry in getattr(parsed_feed, "entries", []) or []:
            title = _entry_field(entry, "title")
            url = _entry_field(entry, "link")
            if not title or not url:
                continue

            published_at = _entry_published(entry)
            if published_at and published_at < start_date:
                continue

            articles.append(
                Article(
                    title=title,
                    url=url,
                    source=feed.name,
                    published_at=published_at,
                    hn_points=None,
                    aggregator=feed.aggregator,
                )
            )

    logging.info(f"Fetched a total of {len(articles)} articles.")
    return articles


def _entry_field(entry: Any, name: str) -> str | None:
    value = getattr(entry, name, None)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _entry_published(entry: Any) -> datetime | None:
    """Read a feed timestamp as UTC.

    feedparser hands back a UTC ``struct_time``; ``timegm`` interprets it as UTC
    where ``mktime`` would apply the local offset and skew every article.
    """
    for field in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, field, None)
        if parsed:
            return datetime.fromtimestamp(calendar.timegm(parsed), tz=UTC)
    return None


def extract_content(
    articles: list[Article],
    download: Callable[[str], str] | None = None,
    delay: float = 0.5,
    max_workers: int = 8,
) -> list[Article]:
    """Extract the body text of each article.

    Downloads run concurrently across hosts but are spaced by ``delay`` seconds
    per host, so the run stays polite without serialising hundreds of fetches.
    Articles are never dropped: an empty body is data the later stages handle,
    and dropping it would silently shrink a story's source count.
    """
    download = download or _download_article_text
    limiter = _HostRateLimiter(delay)

    def extract(article: Article) -> None:
        try:
            article.content = limiter.run(article.url, download) or None
        except Exception as e:  # noqa: BLE001 - a paywall must not end the run
            logging.warning(f"Failed to extract content from {article.url}: {e}")
            article.content = None

    if articles:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(extract, articles))

    extracted = sum(1 for a in articles if a.content)
    logging.info(f"Extracted content for {extracted}/{len(articles)} articles.")
    return articles


class _HostRateLimiter:
    """Serialise requests per host and space them by ``delay`` seconds."""

    def __init__(self, delay: float):
        self._delay = delay
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._last: dict[str, float] = {}

    def run(self, url: str, download: Callable[[str], str]) -> str:
        host = urlsplit(url).netloc.lower()
        with self._guard:
            lock = self._locks.setdefault(host, threading.Lock())

        with lock:
            wait = self._delay - (time.monotonic() - self._last.get(host, -self._delay))
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()
            return download(url)


def _download_article_text(url: str) -> str:
    """Fetch a page and return its article text."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    # Bytes rather than ``response.text``: the document's own declared encoding
    # is more reliable than the header requests guesses from.
    return extract_article_text(response.content, url)


# One parse at a time, process-wide. Downloads overlap freely in the thread
# pool — the network is where the minutes go — but the parses they hand to
# trafilatura do not: concurrent libxml2 parses corrupted the heap on the
# Linux runners ("free(): invalid pointer" on 2026-08-31 at one point in the
# extract stage, "double free or corruption" on the retry at another — a
# moving crash site is a race, not a document). macOS never reproduced it;
# its libxml2 is a different build. Serialising only the parse keeps the
# whole stage's wall clock within seconds of the concurrent version.
_PARSE_LOCK = threading.Lock()


def extract_article_text(html: str | bytes, url: str | None = None) -> str:
    """Extract the article body from a downloaded page.

    Separate from the download so the parsing rules can be tested against fixed
    HTML rather than the live web. Returns an empty string for anything that is
    not an article — a PDF, a plain text file, a JavaScript-rendered shell —
    which the caller stores as no content rather than treating as a failure.
    """
    try:
        with _PARSE_LOCK:
            body = trafilatura.extract(html, url=url, **_EXTRACT_OPTIONS) or ""
            pruned = (
                trafilatura.extract(
                    html, url=url, prune_xpath=_RECIRCULATION_XPATH, **_EXTRACT_OPTIONS
                )
                or ""
            )
    except Exception as e:  # noqa: BLE001 - one malformed page must not end the run
        logging.warning(f"Failed to parse {url}: {e}")
        return ""

    if len(pruned) >= _MIN_PRUNED_BODY or len(pruned) >= len(body):
        return pruned
    return body


def cluster_articles(articles: list[Article]) -> list[Story]:
    """Cluster articles covering the same event into stories.

    Articles are visited in a deterministic order so the same input always
    produces the same clusters, and a match needs more than a high
    ``token_set_ratio``: that measure scores a strict subset at 100, which would
    merge "Apple" into every Apple headline of the week.
    """
    ordered = sorted(articles, key=lambda a: (a.published_at or datetime.max, a.url))
    index = _HeadlineIndex(a.title for a in ordered)

    stories: list[Story] = []
    for article in ordered:
        match = next((s for s in stories if index.same_story(article.title, s.title)), None)
        if match:
            match.articles.append(article)
        else:
            stories.append(Story(title=article.title, articles=[article]))

    for story in stories:
        prominent = _representative_article(story)
        if prominent:
            story.title = prominent.title

    logging.info(f"Clustered {len(articles)} articles into {len(stories)} stories.")
    return stories


# Tuned by replaying a measured live run of 508 real articles and reading every
# merge it produced.
#
# The fuzzy ratios alone could not do this job. Outlets rewrite a wire story
# hard: the four headlines about Anthropic watermarking Claude scored 60-83 on
# token_set and 45-81 on token_sort, so the old token_set >= 80 gate found 2
# multi-source clusters in the whole week. Dropping the ratios far enough to
# catch them let unrelated stories in, because a long headline accumulates
# accidental word overlap.
#
# What actually separates "same event" from "same beat" is *which* words two
# headlines share. Shared rare words ("watermark", "ringcentral") are evidence;
# shared common ones ("samsung", "windows", "google") are not, and this batch of
# headlines is its own corpus for deciding which is which. So the real gate is
# an inverse-document-frequency-weighted overlap, and token_set survives only as
# a cheap sanity floor.
TITLE_SET_THRESHOLD = 60
# Four shared keywords, not three: at three, "Samsung Galaxy Z Fold 8 Ultra
# review" merged with "The Best Samsung Galaxy S26 Cases" on {samsung, galaxy,
# ultra}. Four costs one true merge a week and buys zero false ones.
MIN_SHARED_KEYWORDS = 4
MIN_KEYWORD_OVERLAP = 0.5
# Two headlines this close are the same string with punctuation moved, which no
# keyword count can be asked to prove: "Qwen3.8 27B" and "Qwen3.8-27B" share one
# significant word between them.
NEAR_VERBATIM_SORT_THRESHOLD = 90

# Words too common in headlines to be evidence that two articles share a subject.
_HEADLINE_STOPWORDS = frozenset({
    "about",
    "after",
    "again",
    "against",
    "best",
    "could",
    "first",
    "from",
    "have",
    "into",
    "more",
    "most",
    "new",
    "news",
    "over",
    "says",
    "some",
    "still",
    "that",
    "their",
    "them",
    "then",
    "this",
    "what",
    "when",
    "will",
    "with",
    "your",
})


class _HeadlineIndex:
    """Scores headline pairs against the vocabulary of one batch of headlines.

    Not a global: the weighting is only meaningful relative to the week it was
    built from, so it is constructed per call to :func:`cluster_articles`.
    """

    def __init__(self, titles: Iterable[str]):
        self._document_frequency: Counter[str] = Counter()
        count = 0
        for title in titles:
            count += 1
            self._document_frequency.update(_keywords(title))
        self._documents = max(count, 1)

    def _idf(self, word: str) -> float:
        """Rarity of ``word`` in this batch.

        Smoothed as ``log(1 + N/(1+df))`` rather than the textbook ``log(N/df)``
        so it stays positive: the plain form goes negative once a word appears
        in most documents, which turns a two-headline batch into nonsense.
        """
        return math.log(1 + self._documents / (1 + self._document_frequency[word]))

    def _weight(self, words: set[str]) -> float:
        return sum(self._idf(word) for word in words)

    def same_story(self, left: str, right: str) -> bool:
        """Decide whether two headlines describe the same event."""
        if fuzz.token_sort_ratio(left, right) >= NEAR_VERBATIM_SORT_THRESHOLD:
            return True

        left_words, right_words = _keywords(left), _keywords(right)
        shared = left_words & right_words
        if len(shared) < MIN_SHARED_KEYWORDS:
            return False

        # Normalise by the shorter headline, not the union: a wire story and a
        # 30-word Techmeme summary of it should still read as one event.
        denominator = min(self._weight(left_words), self._weight(right_words))
        if denominator <= 0 or self._weight(shared) / denominator < MIN_KEYWORD_OVERLAP:
            return False

        return fuzz.token_set_ratio(left, right) >= TITLE_SET_THRESHOLD


def _keywords(title: str) -> set[str]:
    """Significant words of a headline, crudely singularised.

    Bare numbers are dropped: six WIRED coupon posts shared {promo, code,
    august, 2026} and merged into one "story" purely on the year. Plurals are
    collapsed so "watermark" and "watermarks" count as the same evidence.
    """
    words = re.findall(r"[a-z0-9]+", title.lower())
    return {
        _singular(w)
        for w in words
        if len(w) > 3 and w not in _HEADLINE_STOPWORDS and not w.isdigit()
    }


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def dedupe_articles(articles: list[Article]) -> list[Article]:
    """Collapse articles that point at the same document.

    The same story reaches us from several feeds (and from Hacker News), often
    with tracking parameters appended. Keeping one entry per canonical URL stops
    a single article from inflating a story's source count.
    """
    by_url: dict[str, Article] = {}
    for article in articles:
        key = normalize_url(article.url)
        if not key:
            continue

        existing = by_url.get(key)
        if existing is None:
            by_url[key] = article
            continue

        # Merge: keep whichever copy carries the richer data.
        if not existing.content and article.content:
            article.hn_points = _max_points(existing, article)
            by_url[key] = article
        else:
            existing.hn_points = _max_points(existing, article)

    logging.info(f"Deduplicated {len(articles)} articles into {len(by_url)} unique URLs.")
    return list(by_url.values())


def _max_points(*articles: Article) -> int | None:
    points = [a.hn_points for a in articles if a.hn_points is not None]
    return max(points) if points else None


# Two headlines this close describe the same event even when the outlets that
# ran them chose different links, which is how most repeats actually arrive.
REPEAT_TITLE_SIMILARITY = 85


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


def _edition_date(week: Any) -> date | None:
    if not isinstance(week, str):
        return None
    try:
        return datetime.strptime(week, "%Y-%m-%d").date()
    except ValueError:
        return None


def _was_published(story: Story, urls: set[str], titles: list[str]) -> bool:
    if any(normalize_url(a.url) in urls for a in story.articles if a.url):
        return True
    return any(
        fuzz.token_sort_ratio(story.title, title) >= REPEAT_TITLE_SIMILARITY for title in titles
    )


def score_stories(stories: list[Story], config: Config, now: datetime | None = None) -> list[Story]:
    """Score stories on Hacker News traction, breadth of coverage and recency.

    Each component is normalised to 0..1 before its weight is applied, so the
    weights in ``config.yaml`` express relative importance directly. Breadth is
    counted as *additional* sources: a story only one outlet ran earns nothing
    for coverage, which is what keeps a slow news day from ranking on recency
    alone.
    """
    now = now or datetime.now(UTC)
    window_hours = config.window_days * 24

    for story in stories:
        points = max((a.hn_points or 0) for a in story.articles) if story.articles else 0
        hn_score = min(points, config.hn.points_cap) / config.hn.points_cap * config.weights.hn

        sources = {a.source for a in story.articles}
        src_score = max(len(sources) - 1, 0) * config.weights.src

        dates = [a.published_at for a in story.articles if a.published_at]
        age_hours = (now - max(dates)).total_seconds() / 3600 if dates else 0.0
        rec_score = _clamp01(1 - age_hours / window_hours) * config.weights.rec

        story.score = hn_score + src_score + rec_score

    return sorted(stories, key=lambda s: s.score, reverse=True)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def categorize_stories(stories: list[Story]) -> list[Story]:
    """Assign each story exactly one category from ``text.CATEGORIES``.

    Matching is word-bounded and considers the extracted body as well as the
    headline: a plain substring search over titles alone labelled "Britain",
    "train", "Hearing Aids" and "email" as AI stories.
    """
    for story in stories:
        story.category = _classify_story(story)

    counts = Counter(s.category for s in stories)
    logging.info(f"Categorized {len(stories)} stories: {dict(counts)}")
    return stories


def _classify_story(story: Story) -> str:
    """Pick a story's category, headlines first and in precedence order.

    Both passes sweep the whole precedence list before falling through, so an
    AI story is never filed as Security just because the Security regex
    happened to match a different article in the same cluster. Headlines are
    decisive; a body only speaks when no headline in the cluster did, and then
    only above a mention threshold, because one passing mention of "AI" in a
    long article does not make the article about AI.
    """
    titles = [story.title, *(a.title for a in story.articles)]
    for category in CATEGORY_PRECEDENCE:
        if any(contains_category_terms(title, category) for title in titles):
            return category

    for category in CATEGORY_PRECEDENCE:
        mentions = max(
            (count_category_terms(a.content, category) for a in story.articles),
            default=0,
        )
        if mentions >= BODY_MENTION_THRESHOLD:
            return category

    return GENERAL_CATEGORY


def select_top_stories(
    stories: list[Story],
    count: int = 7,
    min_ai: int = 4,
    max_per_source: int = 0,
) -> list[Story]:
    """Select the top ``count`` stories, guaranteeing a floor of AI coverage.

    Two preferences temper the raw ranking, and both yield rather than shorten
    an edition. Stories with an extracted body come first, because a story with
    no body cannot be summarized and one too many of those fails the publish
    gate — which is how the 2026-08-24 run died. And no outlet may hold more
    than ``max_per_source`` slots, because whoever supplies the strongest
    signal otherwise owns the page.

    The AI floor is a minimum, not a quota: a week where every top story is
    about AI publishes seven AI stories. Promotion only happens when the
    ranking falls short, and it displaces the weakest general stories rather
    than reordering the digest.
    """
    ranked = sorted(stories, key=lambda s: s.score, reverse=True)
    preferred = [s for s in ranked if _has_body(s)] + [s for s in ranked if not _has_body(s)]
    selected = _pick_with_source_cap(preferred, count, max_per_source)
    selected = _enforce_ai_floor(selected, preferred, min_ai)
    return sorted(selected, key=lambda s: s.score, reverse=True)


def select_edition(
    candidates: list[Story],
    verdict: "editorial.EditorVerdict",
    count: int = 7,
    min_ai: int = 4,
    max_per_source: int = 0,
) -> list[Story]:
    """Turn the editor's verdict into the edition, guards applied.

    The editor proposes; the guards dispose. Its picks lead, in print order,
    but they pass through the same rules as the mechanical path — bodyless
    stories yield to summarizable ones, no outlet exceeds the cap, and the AI
    floor holds — with the score ranking backfilling whatever the guards
    remove. The editor can reorder the page, not overrule its constraints.
    """
    picks = []
    for pick in verdict.picks:
        story = candidates[pick.n - 1]
        story.why = pick.why or None
        picks.append(story)

    chosen = {id(s) for s in picks}
    pool = sorted(
        (s for s in candidates if id(s) not in chosen), key=lambda s: s.score, reverse=True
    )
    preferred = (
        [s for s in picks if _has_body(s)]
        + [s for s in pool if _has_body(s)]
        + [s for s in picks if not _has_body(s)]
        + [s for s in pool if not _has_body(s)]
    )
    selected = _pick_with_source_cap(preferred, count, max_per_source)
    return _enforce_ai_floor(selected, preferred, min_ai)


def _enforce_ai_floor(selected: list[Story], preferred: list[Story], min_ai: int) -> list[Story]:
    """Promote AI stories from ``preferred`` until the floor holds.

    Promotion displaces the weakest general stories — the last non-AI entries
    of a selection that arrives best-first — and never runs when the pool has
    nothing to promote.
    """
    shortfall = min_ai - sum(s.category == AI_CATEGORY for s in selected)
    if shortfall <= 0:
        return selected

    chosen = {id(s) for s in selected}
    promoted = [s for s in preferred if id(s) not in chosen and s.category == AI_CATEGORY][
        :shortfall
    ]
    if promoted:
        demoted = {
            id(s) for s in [s for s in selected if s.category != AI_CATEGORY][-len(promoted) :]
        }
        selected = [s for s in selected if id(s) not in demoted] + promoted
        logging.info(f"Promoted {len(promoted)} AI stories to meet the coverage floor.")
    return selected


def _has_body(story: Story) -> bool:
    return any(a.content for a in story.articles)


def _story_source(story: Story) -> str:
    article = _representative_article(story)
    return article.source if article else ""


def _pick_with_source_cap(preferred: list[Story], count: int, max_per_source: int) -> list[Story]:
    """Take stories in order, deferring any outlet already at the cap.

    Deferred stories return, best first, when the pool cannot otherwise fill
    the edition: the cap trades slots between outlets, never for empty ones.
    """
    if max_per_source < 1:
        return preferred[:count]

    selected: list[Story] = []
    deferred: list[Story] = []
    held = Counter[str]()
    for story in preferred:
        if len(selected) == count:
            break
        source = _story_source(story)
        if held[source] < max_per_source:
            held[source] += 1
            selected.append(story)
        else:
            deferred.append(story)

    if len(selected) < count and deferred:
        backfilled = deferred[: count - len(selected)]
        logging.info(f"Source cap yielded: backfilled {len(backfilled)} over-cap stories.")
        selected += backfilled
    return selected


def summarize_stories(
    stories: list[Story],
    summarizer: Summarizer,
    metrics: RunMetrics | None = None,
) -> list[Story]:
    """Generate a summary for each story.

    Every article in the cluster is a candidate, richest body first: summarizing
    only the first article meant one paywalled outlet sank the whole story.
    ``summary`` is left unset when nothing works, so the publish gate can see it.
    """
    record = metrics if metrics is not None else RunMetrics()

    for story in stories:
        # Original reporting first, richest body first. An aggregator's page is
        # a list of other outlets' coverage, and asking a model to summarize
        # one produced a refusal on the live page.
        candidates = sorted(
            (a for a in story.articles if a.content),
            key=lambda a: (not a.aggregator, len(a.content or "")),
            reverse=True,
        )
        story.summary = None
        for article in candidates:
            summary = summarizer.summarize(article.content or "")
            if summary:
                story.summary = summary
                # Read the model straight after the call that answered, so an
                # implementation reporting its fallback is captured correctly.
                record.summary_models[story.title] = summarizer_model(summarizer)
                break
        if not story.summary:
            logging.warning(f"No summary produced for: {story.title}")

    record.summaries = sum(1 for s in stories if s.summary)
    record.summaries_failed = len(stories) - record.summaries
    logging.info(f"Generated summaries for {record.summaries}/{len(stories)} stories.")
    return stories


def build_digest(
    config: Config,
    summarizer: Summarizer,
    now: datetime | None = None,
    parse: Callable[[str], Any] | None = None,
    download: Callable[[str], str] | None = None,
    hn_fetch: hn.JsonFetcher | None = None,
    search: discovery.SearchFn | None = None,
    editor: "editorial.Editor | None" = None,
    delay: float = 0.5,
    editions: list[dict[str, Any]] | None = None,
    metrics: RunMetrics | None = None,
) -> Digest:
    """Run every curation stage and return the stories to publish.

    Bodies are downloaded only for the candidate pool, after ranking: fetching
    every article of the week costs hundreds of requests to decide seven slots.

    ``editions`` is the published archive, used to keep a story that is still
    hot from running twice; the caller reads it so this stays free of the file
    system. ``metrics`` is filled in as the stages run — pass one to keep the
    record, or leave it unset and the run throws its measurements away.
    """
    now = now or datetime.now(UTC)
    record = metrics if metrics is not None else RunMetrics()

    with record.stage("fetch"):
        feed_articles = fetch_articles(config, now=now, parse=parse)
        hn_articles = hn.fetch_hn_articles(config, now=now, fetch=hn_fetch)
    record.articles_from_feeds = len(feed_articles)
    record.articles_from_hn = len(hn_articles)
    record.articles_before_dedupe = len(feed_articles) + len(hn_articles)

    with record.stage("dedupe"):
        articles = dedupe_articles(feed_articles + hn_articles)
        hn.merge_hn_points(articles, hn_articles)
    record.articles_after_dedupe = len(articles)

    with record.stage("cluster"):
        stories = cluster_articles(articles)
    record.stories = len(stories)
    source_counts = [len({a.source for a in s.articles}) for s in stories]
    record.multi_source_stories = sum(1 for c in source_counts if c > 1)
    record.max_sources = max(source_counts, default=0)

    with record.stage("score"):
        stories = score_stories(stories, config, now=now)

    with record.stage("consensus"):
        consensus = discovery.fetch_consensus(config, now=now, search=search)
        missed = discovery.apply_consensus_boost(stories, consensus, config.weights.consensus)
        if consensus:
            stories.sort(key=lambda s: s.score, reverse=True)
    record.consensus_stories = len(consensus)
    record.consensus_matched = len(consensus) - len(missed)
    record.consensus_missed = [m.headline for m in missed]

    with record.stage("exclude_repeats"):
        ranked = drop_recently_published(
            stories,
            editions or [],
            now=now,
            lookback=timedelta(weeks=config.digest.repeat_lookback_weeks),
            keep_at_least=config.digest.story_count,
        )
    record.repeats_dropped = len(stories) - len(ranked)

    candidates = ranked[: config.digest.candidate_pool]
    record.candidate_stories = len(candidates)

    with record.stage("extract"):
        candidate_articles = [a for s in candidates for a in s.articles]
        extract_content(candidate_articles, download=download, delay=delay)
    record.record_extraction(candidate_articles)

    with record.stage("categorize"):
        candidates = categorize_stories(candidates)
    record.categories = dict(Counter(s.category for s in candidates))

    with record.stage("select"):
        # The natural count is measured before selection so the record can say
        # whether the ranking met the AI floor on its own or the floor overrode
        # it. select_top_stories only logs that it promoted, not from what.
        by_score = sorted(candidates, key=lambda s: s.score, reverse=True)
        record.ai_before_promotion = sum(
            s.category == AI_CATEGORY for s in by_score[: config.digest.story_count]
        )
        verdict = None
        if editor is not None:
            verdict = editor.select(
                candidates,
                count=config.digest.story_count,
                min_ai=config.digest.min_ai_stories,
                max_per_source=config.digest.max_per_source,
            )
        if editor is not None and verdict is not None:
            selected = select_edition(
                candidates,
                verdict,
                count=config.digest.story_count,
                min_ai=config.digest.min_ai_stories,
                max_per_source=config.digest.max_per_source,
            )
            record.editor_used = True
            record.editor_model = editor.last_model or ""
            record.editor_completion_tokens = editor.last_completion_tokens
            record.editor_reasoning_tokens = editor.last_reasoning_tokens
            record.editor_max_tokens = editor.max_tokens
            picked = {id(candidates[p.n - 1]) for p in verdict.picks}
            record.editor_overridden = sum(1 for s in selected if id(s) not in picked)
        else:
            selected = select_top_stories(
                candidates,
                count=config.digest.story_count,
                min_ai=config.digest.min_ai_stories,
                max_per_source=config.digest.max_per_source,
            )
    record.ai_published = sum(s.category == AI_CATEGORY for s in selected)
    record.ai_promoted = max(record.ai_published - record.ai_before_promotion, 0)

    with record.stage("summarize"):
        summarized = summarize_stories(selected, summarizer, metrics=record)
    return Digest(stories=summarized, intro=(verdict.intro or None) if verdict else None)


def build_edition(
    stories: list[Story],
    week: str,
    now: datetime | None = None,
    intro: str | None = None,
) -> dict[str, Any]:
    """Assemble the published JSON payload for one week."""
    now = now or datetime.now(UTC)
    entries = []

    for rank, story in enumerate(stories, start=1):
        main_article = _representative_article(story)
        sources = sorted({a.source for a in story.articles})
        entries.append({
            "rank": rank,
            "title": story.title,
            "source": main_article.source if main_article else "",
            "sources": sources,
            "source_count": len(sources),
            "url": main_article.url if main_article else "",
            "category": story.category,
            "hn_points": main_article.hn_points if main_article else None,
            "score": round(story.score, 3),
            "summary": story.summary or "",
            "why": story.why or "",
            "consensus": story.consensus,
        })

    return {
        "week": week,
        "generated_at": now.isoformat(),
        "intro": intro or "",
        "stories": entries,
    }


def _representative_article(story: Story) -> Article | None:
    """Pick the article to link to: original reporting over an aggregator's
    rewrite, then the most traction, then the fullest body."""
    if not story.articles:
        return None
    return max(
        story.articles,
        key=lambda a: (not a.aggregator, a.hn_points or 0, len(a.content or "")),
    )


def save_edition(edition: dict[str, Any], data_dir: Path) -> tuple[Path, Path]:
    """Write the edition to ``latest.json`` and to a permanent archive copy."""
    archive_dir = data_dir / ARCHIVE_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)

    latest_path = data_dir / "latest.json"
    archive_path = archive_dir / f"{edition['week']}.json"
    payload = json.dumps(edition, indent=2, ensure_ascii=False) + "\n"

    latest_path.write_text(payload, encoding="utf-8")
    archive_path.write_text(payload, encoding="utf-8")
    logging.info(f"Saved edition to {latest_path} and {archive_path}")
    return latest_path, archive_path


def generate_site(
    editions: list[dict[str, Any]],
    output_dir: Path,
    template_dir: Path | None = None,
    static_dir: Path | None = None,
    site_url: str = DEFAULT_SITE_URL,
) -> list[Path]:
    """Render the digest as a static site: the latest edition plus an archive.

    Paths default to the installed package, so the pipeline works from any
    working directory rather than only the repository root.
    """
    if not editions:
        logging.warning("No editions to render; skipping site generation.")
        return []

    template_dir = template_dir or PACKAGE_DIR / "templates"
    static_dir = static_dir or PACKAGE_DIR / "static"
    editions = sorted(editions, key=lambda e: e["week"], reverse=True)

    env = Environment(
        loader=FileSystemLoader(template_dir),
        # Unconditional: select_autoescape() keys off the file extension and
        # silently left ".html.jinja" templates unescaped, so feed-supplied
        # titles were rendered as markup.
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    template = env.get_template("edition.html.jinja")

    latest, past = editions[0], editions[1:]
    written = [
        _render_page(
            template,
            output_dir / "index.html",
            latest,
            past,
            root="",
            is_latest=True,
            site_url=site_url,
        )
    ]

    archive_dir = output_dir / ARCHIVE_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)
    for edition in editions:
        written.append(
            _render_page(
                template,
                archive_dir / f"{edition['week']}.html",
                edition,
                [e for e in editions if e["week"] != edition["week"]],
                root="../",
                is_latest=False,
                site_url=site_url,
            )
        )

    written.extend(syndication.write_syndication(editions, output_dir, site_url))

    for asset in sorted(static_dir.glob("*")):
        if asset.is_file():
            shutil.copy(asset, output_dir / asset.name)
    logging.info(f"Generated {len(written)} pages in {output_dir}")
    return written


def _render_page(
    template: Any,
    path: Path,
    edition: dict[str, Any],
    past_editions: list[dict[str, Any]],
    root: str,
    is_latest: bool,
    site_url: str = DEFAULT_SITE_URL,
) -> Path:
    week = edition["week"]
    page_title = "LastWeekIn.Tech" if is_latest else f"LastWeekIn.Tech — week ending {week}"
    html = template.render(
        edition=edition,
        past_editions=past_editions,
        root=root,
        is_latest=is_latest,
        site_url=site_url,
        page_title=page_title,
        description=(
            f"The {len(edition['stories'])} most important tech and AI stories "
            f"of the week ending {week}, summarized with links to the original reporting."
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def list_editions(data_dir: Path) -> list[dict[str, Any]]:
    """Return every archived edition, newest first."""
    archive_dir = data_dir / ARCHIVE_DIRNAME
    if not archive_dir.is_dir():
        return []

    editions = []
    for path in sorted(archive_dir.glob("*.json")):
        try:
            edition = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Skipping unreadable archive entry {path}: {e}")
            continue
        if isinstance(edition, dict) and edition.get("week"):
            editions.append(edition)

    return sorted(editions, key=lambda e: e["week"], reverse=True)
