"""
Configuration loading for the LastWeekIn.Tech pipeline.
"""

import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Matches a whole-value environment reference such as "${HF_TOKEN}".
_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class ConfigError(ValueError):
    """Raised when the configuration file is missing or malformed."""


@dataclass
class Feed:
    """Represents an RSS feed source."""

    name: str
    url: str
    # An aggregator reports on coverage rather than producing it. Its presence
    # in a cluster counts as corroboration, but it is never the linked source.
    aggregator: bool = False


@dataclass
class HNSettings:
    """Represents Hacker News settings."""

    min_points: int = 50
    points_cap: int = 500
    limit: int = 200
    enabled: bool = True


@dataclass
class Weights:
    """Represents scoring weights."""

    hn: float = 5.0
    src: float = 3.0
    rec: float = 1.0
    # Flat bonus for a story the Perplexity consensus check also carries.
    # Unlike the others it is not normalised — it is corroboration, applied
    # once per story after the base score.
    consensus: float = 2.0


@dataclass
class EditorSettings:
    """The editorial selection stage: one model call that picks the edition."""

    enabled: bool = True
    model_name: str = "anthropic/claude-sonnet-5"
    fallback_models: list[str] = field(
        default_factory=lambda: ["anthropic/claude-haiku-4.5", "google/gemini-3.7-flash"]
    )
    # Generous on purpose: reasoning-capable models spend part of this budget
    # thinking, and a budget sized for the JSON alone truncated the primary
    # model's verdict mid-answer on the first live run.
    max_tokens: int = 8000
    temperature: float = 0.2
    # How many characters of each candidate's body the editor reads.
    excerpt_chars: int = 300


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


@dataclass
class PerplexitySettings:
    """The consensus check against the wider press, via the Perplexity API."""

    enabled: bool = True
    # Unset disables the stage rather than failing the run.
    api_key: str | None = None
    model: str = "sonar-pro"
    # How many consensus stories to ask for.
    story_count: int = 15


@dataclass
class DigestSettings:
    """Represents the shape of a published edition."""

    story_count: int = 7
    min_ai_stories: int = 4
    max_missing_summaries: int = 1
    # Summaries that exist but fail a quality check (truncated mid-sentence,
    # never naming their subject, quoting a figure absent from the source).
    # One weak summary is a bad week; a digest of them is not worth publishing.
    max_low_quality_summaries: int = 2
    # How many top-scoring stories stay in contention. Only these have their
    # article bodies downloaded, which is the slowest stage of the run.
    candidate_pool: int = 40
    # How far back to look for stories we already published. Hacker News
    # traction persists for days, so a story hot for eight days can top two
    # consecutive editions. Zero or less publishes repeats.
    repeat_lookback_weeks: int = 3
    # How many stories one outlet may place in an edition. Without a cap the
    # strongest signal owns the page: WIRED supplied 40% of every story the old
    # ranking ever published, and the first HN-scored edition was seven of
    # seven Hacker News. Zero disables the cap; it always yields rather than
    # shorten an edition.
    max_per_source: int = 2


@dataclass
class SummarizerSettings:
    """Represents summarizer settings."""

    model_name: str
    fallback_models: list[str] = field(default_factory=list)
    hf_token: str | None = None
    huggingface_models: list[str] = field(default_factory=list)
    max_tokens: int = 400
    max_input_chars: int = 12000
    temperature: float = 0.3


def _build_ai_daily(data: dict[str, Any]) -> AiDailySettings:
    """Construct AiDailySettings, building its nested dataclass fields the
    same way Config.from_yaml builds its own — a dict for any of ``hn``,
    ``weights`` or ``briefing`` becomes the matching settings object."""
    kwargs = dict(data)
    if "hn" in kwargs:
        default_hn = HNSettings(min_points=20, points_cap=300, limit=200)
        hn_data = {**dataclasses.asdict(default_hn), **kwargs["hn"]}
        kwargs["hn"] = HNSettings(**hn_data)
    if "weights" in kwargs:
        default_weights = Weights()
        weights_data = {**dataclasses.asdict(default_weights), **kwargs["weights"]}
        kwargs["weights"] = Weights(**weights_data)
    if "briefing" in kwargs:
        default_briefing = AiBriefingSettings()
        briefing_data = {**dataclasses.asdict(default_briefing), **kwargs["briefing"]}
        kwargs["briefing"] = AiBriefingSettings(**briefing_data)
    return AiDailySettings(**kwargs)


@dataclass
class Config:
    """Represents the main configuration."""

    feeds: list[Feed]
    hn: HNSettings
    weights: Weights
    window_days: int
    summarizer: SummarizerSettings
    digest: DigestSettings = field(default_factory=DigestSettings)
    perplexity: PerplexitySettings = field(default_factory=PerplexitySettings)
    editor: EditorSettings = field(default_factory=EditorSettings)
    ai_daily: AiDailySettings = field(default_factory=AiDailySettings)
    # Absolute origin of the published site. Feeds, sitemaps and social cards
    # all need absolute URLs, so this cannot be derived from the output path.
    site_url: str = "https://lastweekin.tech"

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Loads the configuration from a YAML file, expanding ``${VAR}`` references."""
        try:
            with open(path) as f:
                data = expand_env(yaml.safe_load(f))
        except FileNotFoundError as exc:
            raise ConfigError(f"Configuration file not found: {path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ConfigError(f"{path} must contain a YAML mapping.")

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
        if unknown:
            raise ConfigError(f"Invalid configuration in {path}: unknown keys {sorted(unknown)}")

        try:
            config = cls(
                feeds=[Feed(**feed) for feed in data.get("feeds", [])],
                hn=HNSettings(**data.get("hn", {})),
                weights=Weights(**data.get("weights", {})),
                window_days=int(data.get("window_days", 7)),
                summarizer=SummarizerSettings(**data["summarizer"]),
                digest=DigestSettings(**data.get("digest", {})),
                perplexity=PerplexitySettings(**data.get("perplexity", {})),
                editor=EditorSettings(**data.get("editor", {})),
                ai_daily=_build_ai_daily(data.get("ai_daily", {})),
                site_url=str(data.get("site_url") or "https://lastweekin.tech").rstrip("/"),
            )
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"Invalid configuration in {path}: {exc}") from exc

        config.validate()
        return config

    def validate(self) -> None:
        """Fail fast on values that would silently produce a broken digest."""
        problems = []
        if not self.feeds:
            problems.append("no feeds configured")
        if self.window_days < 1:
            problems.append("window_days must be at least 1")
        if self.digest.story_count < 1:
            problems.append("digest.story_count must be at least 1")
        if self.digest.min_ai_stories > self.digest.story_count:
            problems.append("digest.min_ai_stories cannot exceed digest.story_count")
        if self.hn.points_cap < 1:
            problems.append("hn.points_cap must be at least 1")
        if not self.summarizer.model_name:
            problems.append("summarizer.model_name is required")
        if self.ai_daily.story_count < 1:
            problems.append("ai_daily.story_count must be at least 1")
        if self.ai_daily.min_story_count > self.ai_daily.story_count:
            problems.append("ai_daily.min_story_count cannot exceed ai_daily.story_count")
        if self.ai_daily.candidate_pool < self.ai_daily.story_count:
            problems.append("ai_daily.candidate_pool must be at least ai_daily.story_count")
        if not self.ai_daily.briefing.model_name:
            problems.append("ai_daily.briefing.model_name is required")
        if not self.site_url.startswith(("http://", "https://")):
            problems.append("site_url must be an absolute http(s) URL")
        if problems:
            raise ConfigError("; ".join(problems))


def expand_env(value: Any) -> Any:
    """Replace ``${VAR}`` strings with their environment value, recursively.

    An unset variable resolves to ``None`` rather than the literal placeholder,
    so an absent secret disables the feature instead of being sent as a token.
    """
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, str):
        match = _ENV_REF.match(value.strip())
        if match:
            return os.environ.get(match.group(1)) or None
    return value


def get_config(path: Path | None = None) -> Config:
    """Returns the application configuration."""
    return Config.from_yaml(path or Path(__file__).parent / "config.yaml")
