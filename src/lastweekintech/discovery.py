"""Cross-checking the week's ranking against the wider press.

Hacker News points are the pipeline's only direct importance signal, and any
single signal distorts the page toward its own audience: the first HN-scored
edition was seven stories of seven from Hacker News. Perplexity's search
models read the week's coverage across the open web, so asking one for the
week's biggest technology stories yields an independent, editorially flavoured
consensus to weigh against our own ranking.

The stage is corroborative, never generative: a consensus story that matches a
candidate raises that candidate's score, and one that matches nothing is only
*recorded* in the run metrics — a headline the funnel never saw is evidence of
a missing feed, not a story we can publish without an article behind it.

Like every network boundary in this pipeline the search call is injectable,
and the stage disables itself when no API key is configured: a missing secret
costs a signal, never an edition.
"""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlsplit

import requests
from thefuzz import fuzz

from lastweekintech.config import Config
from lastweekintech.domain import Article, Story
from lastweekintech.text import normalize_url

PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
TIMEOUT = 120

# (model, prompt) -> raw model text. The default implementation calls the
# Perplexity API; tests inject canned answers.
SearchFn = Callable[[str, str], str]

PROMPT_TEMPLATE = (
    "You are the news editor of a weekly technology digest. List the {count} "
    "most important technology news stories published between {start} and "
    "{end} (UTC), judged by industry impact and breadth of coverage across "
    "reputable outlets — not by any single community's popularity. Include "
    "major AI, security, policy, hardware, open source and business stories.\n\n"
    "Answer with a JSON array only, no prose before or after. Each element: "
    '{{"headline": "...", "why": "one sentence on why it mattered", '
    '"urls": ["link to original reporting", "..."]}}'
)

# AI Daily's own consensus prompt: same JSON contract (parse_consensus and
# apply_consensus_boost need nothing else), narrower editorial brief. Reuses
# the shared PerplexitySettings/story_count rather than a dedicated section —
# this is a corroboration signal on a smaller pool, not a separate budget.
AI_DAILY_PROMPT_TEMPLATE = (
    "You are the AI news editor of a daily AI-only briefing. List the {count} "
    "most important AI news developments published between {start} and "
    "{end} (UTC). Prioritize events with significant implications for AI "
    "capabilities, major companies, open-source models, regulation and "
    "policy, security and safety, infrastructure and chips, and industry "
    "economics. Avoid low-impact product announcements and stories that are "
    "a rehash of something already widely covered.\n\n"
    "Answer with a JSON array only, no prose before or after. Each element: "
    '{{"headline": "...", "why": "one sentence on why it matters", '
    '"urls": ["link to original reporting", "..."]}}'
)

# A consensus headline is a paraphrase, not a quote, so URL identity is
# decisive and title matching needs both a fuzzy floor and shared substance.
MATCH_TOKEN_SET_RATIO = 70
MATCH_SHARED_KEYWORDS = 3

_STOPWORDS = frozenset(
    "about after against amid announces because before between could from have "
    "into launches major more most news over reportedly says been some their "
    "them this that what when will with your".split()
)


@dataclass
class ConsensusStory:
    """One story the wider press led with, per the search model."""

    headline: str
    why: str = ""
    urls: list[str] = field(default_factory=list)


def fetch_consensus(
    config: Config,
    now: datetime | None = None,
    search: SearchFn | None = None,
    prompt_template: str = PROMPT_TEMPLATE,
) -> list[ConsensusStory]:
    """Ask the search model for the window's consensus top stories.

    Returns an empty list — never raises — when the stage is disabled, the key
    is missing, the request fails or the answer cannot be parsed. This signal
    improves an edition; its absence must not cost one.

    ``prompt_template`` defaults to the weekly digest's own brief; AI Daily
    passes ``AI_DAILY_PROMPT_TEMPLATE`` instead. Both fill the same
    ``{count}``/``{start}``/``{end}`` placeholders and both are read by
    ``parse_consensus`` unmodified — only the editorial framing differs.
    """
    settings = config.perplexity
    if not settings.enabled:
        return []
    if search is None:
        if not settings.api_key:
            logging.info("PERPLEXITY_API_KEY not set; skipping the consensus check.")
            return []
        search = _searcher(settings.api_key)

    now = now or datetime.now()
    prompt = prompt_template.format(
        count=settings.story_count,
        start=(now - timedelta(days=config.window_days)).date().isoformat(),
        end=now.date().isoformat(),
    )

    try:
        answer = search(settings.model, prompt)
    except Exception as e:  # noqa: BLE001 - a lost signal must not end the run
        logging.warning(f"Consensus search failed: {e}")
        return []

    stories = parse_consensus(answer)[: settings.story_count]
    logging.info(f"Consensus check returned {len(stories)} stories.")
    return stories


def _searcher(api_key: str) -> SearchFn:
    def search(model: str, prompt: str) -> str:
        response = requests.post(
            PERPLEXITY_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"] or "")

    return search


def parse_consensus(answer: str) -> list[ConsensusStory]:
    """Read the model's JSON array, tolerating prose and code fences around it.

    Anything unparseable yields an empty list; a malformed element is dropped
    rather than sinking the elements that did parse.
    """
    match = re.search(r"\[.*\]", answer or "", re.DOTALL)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []

    stories = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        headline = str(entry.get("headline") or "").strip()
        if not headline:
            continue
        urls = entry.get("urls")
        stories.append(
            ConsensusStory(
                headline=headline,
                why=str(entry.get("why") or "").strip(),
                urls=[str(u) for u in urls if u] if isinstance(urls, list) else [],
            )
        )
    return stories


def apply_consensus_boost(
    stories: list[Story],
    consensus: list[ConsensusStory],
    weight: float,
) -> list[ConsensusStory]:
    """Raise the score of every story the wider press also led with.

    Each consensus story boosts at most one candidate — its best match — so a
    cluster cannot collect the same corroboration twice. Returns the consensus
    stories that matched nothing, which the caller records: a headline the
    funnel never saw is how a missing feed announces itself.
    """
    missed = []
    for entry in consensus:
        story = _best_match(stories, entry)
        if story is None:
            missed.append(entry)
            continue
        story.consensus = True
        story.score += weight

    matched = len(consensus) - len(missed)
    logging.info(f"Consensus corroborated {matched}/{len(consensus)} stories in the pool.")
    return missed


def stories_from_missed(missed: list[ConsensusStory], weight: float) -> list[Story]:
    """Turn consensus stories the feed pool never saw into real candidates.

    A boost can only raise a story that already exists in the pool — most of
    what a live search corroborates is genuinely absent otherwise, and a
    boost applied to nothing does not make it publishable. Each missed entry
    becomes its own single-article Story, scored at the same flat consensus
    weight a corroborated match receives (it has no HN points or breadth
    signal of its own yet to score on).

    This is still not a shortcut past "no article, no publish": the caller's
    normal extract_content step has to fetch a real body from the citation
    URL before the briefer can pick it, and a missing or unextractable URL
    yields no candidate at all here, same as any other extraction failure.
    """
    stories = []
    for entry in missed:
        url = entry.urls[0] if entry.urls else None
        if not url:
            continue
        source = urlsplit(url).netloc.removeprefix("www.") or "the web"
        article = Article(title=entry.headline, url=url, source=source)
        stories.append(
            Story(title=entry.headline, articles=[article], score=weight, consensus=True)
        )
    return stories


def _best_match(stories: list[Story], entry: ConsensusStory) -> Story | None:
    urls = {key for u in entry.urls if (key := normalize_url(u))}
    best, best_ratio = None, 0
    for story in stories:
        if urls & {key for a in story.articles if (key := normalize_url(a.url))}:
            return story
        ratio = fuzz.token_set_ratio(entry.headline, story.title)
        if (
            ratio >= MATCH_TOKEN_SET_RATIO
            and ratio > best_ratio
            and _shares_substance(entry.headline, story.title)
        ):
            best, best_ratio = story, ratio
    return best


def _shares_substance(left: str, right: str) -> bool:
    return len(_keywords(left) & _keywords(right)) >= MATCH_SHARED_KEYWORDS


def _keywords(title: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", title.lower())
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS and not w.isdigit()}
