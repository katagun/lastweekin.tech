"""
Domain models for the LastWeekIn.Tech pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Article:
    """Represents a single news article."""

    title: str
    url: str
    source: str
    content: str | None = None
    published_at: datetime | None = None

    hn_points: int | None = None
    # True for feeds that report on coverage rather than produce it (Techmeme).
    # Their presence in a cluster is corroboration, but the digest should link
    # to the original reporting, never to the aggregator's rewrite.
    aggregator: bool = False


@dataclass
class Story:
    """Represents a deduplicated news story."""

    title: str
    articles: list[Article] = field(default_factory=list)
    summary: str | None = None
    score: float = 0.0
    category: str = "General Tech"
    # True when the Perplexity consensus check also carried this story.
    consensus: bool = False
    # The editor's one-line case for the story; None when the editor did not
    # run or did not pick this story, and the page renders without it.
    why: str | None = None


@dataclass
class Digest:
    """One week's edition as the pipeline hands it to the publisher."""

    stories: list[Story] = field(default_factory=list)
    # The editor's 2-3 sentence read on the week; None when the editor did
    # not run, and the page renders without it.
    intro: str | None = None


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
