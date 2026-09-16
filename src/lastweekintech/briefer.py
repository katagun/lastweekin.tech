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
        theme valid, the per-source cap respected, and every pick backed by
        actual article text — anything less is not worth publishing, since
        there is no mechanical fallback that can write the analysis a
        rejected pick would have carried.
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
            if verdict is not None:
                cap_ok = _within_source_cap(verdict, candidates, max_per_source)
                text_ok = all(_has_article_text(pick, candidates) for pick in verdict.picks)
                if cap_ok and text_ok:
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
                if not cap_ok:
                    logging.info(
                        f"Briefer: {model}'s verdict exceeds the per-source cap "
                        f"(max {max_per_source})."
                    )
                if not text_ok:
                    logging.info(
                        f"Briefer: {model}'s verdict picked a candidate with no article text."
                    )
            logging.warning(
                f"Briefing model {model} returned an unusable verdict "
                f"(finish_reason={completion.finish_reason!r}): {completion.text[:3000]!r}"
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
        lines.append(f"{n}. {story.title}\n   ({'; '.join(signals)})\n   {excerpt}")
    return "Candidates:\n\n" + "\n\n".join(lines)


def _parse_verdict(answer: str, pool: int, min_count: int, max_count: int) -> BriefVerdict | None:
    """Read the model's JSON, accepting nothing less than a complete, valid briefing.

    Every rejection path logs why: a live run's answer is never available
    after the fact (only a short preview is logged at the call site), so a
    silent ``None`` here is undiagnosable from CI output alone.
    """
    match = re.search(r"\{.*\}", answer or "", re.DOTALL)
    if not match:
        logging.info("Briefer: no JSON object found in the response.")
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logging.info(f"Briefer: response is not valid JSON: {e}")
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("picks"), list):
        logging.info("Briefer: response is not an object with a 'picks' list.")
        return None

    picks = []
    seen = set()
    for entry in raw["picks"]:
        if not isinstance(entry, dict):
            logging.info(f"Briefer: a pick is not an object: {entry!r}")
            return None
        n = entry.get("n")
        theme = entry.get("theme")
        what_happened = str(entry.get("what_happened") or "").strip()
        why_it_matters = str(entry.get("why_it_matters") or "").strip()
        watch_next = str(entry.get("watch_next") or "").strip()

        if not isinstance(n, int) or not 1 <= n <= pool or n in seen:
            logging.info(f"Briefer: pick has an invalid or duplicate n={n!r} (pool size {pool}).")
            return None
        if theme not in THEMES:
            logging.info(f"Briefer: pick n={n} has theme {theme!r}, not one of {THEMES}.")
            return None
        if not what_happened or not why_it_matters or not watch_next:
            logging.info(f"Briefer: pick n={n} is missing what_happened/why_it_matters/watch_next.")
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

    # Trust an over-long list down to the print order, never an under-long
    # one — mirrors editor.py's _parse_verdict. A model asked for "4-5" that
    # answers with 6 is being generous, not wrong; rejecting the whole
    # verdict over it means the fallback chain burns every model for no
    # reason on exactly the days with the most AI news to choose from.
    if len(picks) < min_count:
        logging.info(f"Briefer: only {len(picks)} valid picks, need at least {min_count}.")
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


def _has_article_text(pick: BriefPick, candidates: list[Story]) -> bool:
    return any(a.content for a in candidates[pick.n - 1].articles)
