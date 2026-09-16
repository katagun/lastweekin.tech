"""Tests for AI Daily's briefing step: the Briefer and the guards around its verdict."""

import json

import pytest
from conftest import make_article, make_story

from lastweekintech.briefer import THEMES, Briefer
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

    def test_truncates_an_overlong_list_to_print_order(self):
        # A model asked for "4-5" that answers with 6 is being generous, not
        # wrong — mirrors editor.py's identical tolerance. Observed live: every
        # configured model returned more than max_count on a heavy AI-news day,
        # and the stricter reject-instead-of-truncate behavior burned the whole
        # fallback chain for it.
        briefer = make_briefer(verdict_for(1, 2, 3, 4, 5, 6))
        verdict = briefer.brief(pool(), count=5, min_count=4, max_per_source=0, recent_titles=[])
        assert verdict is not None
        assert [p.n for p in verdict.picks] == [1, 2, 3, 4, 5]

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
            {
                "picks": [
                    {
                        "n": 1,
                        "theme": "Not A Real Theme",
                        "what_happened": "x",
                        "why_it_matters": "y",
                        "watch_next": "z",
                    },
                    pick(2),
                ]
            },
            {
                "picks": [
                    {
                        "n": 1,
                        "theme": THEMES[0],
                        "what_happened": "",
                        "why_it_matters": "y",
                        "watch_next": "z",
                    },
                    pick(2),
                ]
            },
        ],
    )
    def test_an_unusable_verdict_yields_none(self, payload):
        briefer = make_briefer(payload)
        assert (
            briefer.brief(pool(), count=2, min_count=2, max_per_source=0, recent_titles=[]) is None
        )

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

    def test_hacker_news_never_counts_toward_the_source_cap(self):
        # Live production failure: a heavy-HN day had 3+ topically unrelated
        # candidates (a security incident, a startup launch, a policy story)
        # that never got mainstream press pickup, so they all shared "Hacker
        # News" as their only source — and every configured model tripped
        # the cap picking the genuinely best stories of the day. HN is a
        # discovery channel, not an outlet with an editorial share to bound.
        briefer = make_briefer(verdict_for(1, 2, 3))
        verdict = briefer.brief(
            pool(count=3, source="Hacker News"),
            count=3,
            min_count=3,
            max_per_source=1,
            recent_titles=[],
        )
        assert verdict is not None

    def test_rejects_a_verdict_that_picks_a_candidate_with_no_article_text(self):
        # Candidate 2 has no article body, so a verdict picking it must be
        # discarded even though it is otherwise well-formed.
        candidates = pool(count=2)
        candidates[1] = make_story(
            title="story 1",
            score=99,
            articles=[make_article(title="story 1", url="https://x/1", content=None)],
        )
        briefer = make_briefer(verdict_for(1, 2))
        verdict = briefer.brief(
            candidates, count=2, min_count=2, max_per_source=0, recent_titles=[]
        )
        assert verdict is None

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
            pool(),
            count=2,
            min_count=2,
            max_per_source=0,
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

    def test_the_prompt_shows_the_exact_source_the_cap_will_enforce(self):
        # Live models tripped the per-source cap on their first try despite
        # visibly spreading picks across outlets, because the prompt used to
        # list every outlet a multi-source story had ("outlets: A, B, C")
        # while the cap counts one resolved representative source per story
        # — a mismatch a model has no way to predict. The prompt must show
        # exactly the string _within_source_cap will count against.
        from lastweekintech.briefer import _story_source

        multi_source = make_story(
            title="Covered by several outlets",
            articles=[
                make_article(title="Covered by several outlets", source="Ars Technica"),
                make_article(
                    title="Covered by several outlets", source="Hacker News", hn_points=500
                ),
            ],
        )
        seen = {}

        def complete(model, messages, max_tokens):
            seen["prompt"] = messages[1]["content"]
            return Completion(text=json.dumps(verdict_for(1)))

        Briefer(AiBriefingSettings(), complete=complete).brief(
            [multi_source], count=1, min_count=1, max_per_source=0, recent_titles=[]
        )
        assert f"source: {_story_source(multi_source)}" in seen["prompt"]
