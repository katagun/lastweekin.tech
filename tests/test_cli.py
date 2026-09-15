"""Tests for the CLI wiring, especially the publish gate."""

import json

import pytest
from conftest import make_article, make_story
from typer.testing import CliRunner

from lastweekintech import main

runner = CliRunner()


@pytest.fixture
def stub_pipeline(monkeypatch, config):
    """Run the CLI against a canned digest instead of the network."""

    def stories(count=7, missing=0):
        built = []
        for i in range(count):
            story = make_story(
                title=f"Story {i}",
                score=100 - i,
                articles=[make_article(url=f"https://example.com/{i}")],
            )
            story.summary = None if i < missing else "A perfectly serviceable summary of news."
            built.append(story)
        return built

    state = {"stories": stories()}
    monkeypatch.setattr(main, "get_config", lambda path=None: config)
    monkeypatch.setattr(main, "Summarizer", lambda settings: object())
    monkeypatch.setattr(main, "Editor", lambda settings: object())
    monkeypatch.setattr(
        main.pipeline,
        "build_digest",
        lambda *a, **k: main.pipeline.Digest(stories=state["stories"]),
    )
    state["make"] = stories
    return state


def invoke(tmp_path, *args):
    return runner.invoke(
        main.app,
        ["run", "--data-dir", str(tmp_path / "data"), "--site-dir", str(tmp_path), *args],
    )


class TestRun:
    def test_publishes_a_good_digest(self, tmp_path, stub_pipeline):
        result = invoke(tmp_path, "--week", "2026-08-10")
        assert result.exit_code == 0
        assert (tmp_path / "index.html").exists()
        assert (tmp_path / "data" / "archive" / "2026-08-10.json").exists()

    def test_refuses_to_publish_a_broken_digest(self, tmp_path, stub_pipeline):
        stub_pipeline["stories"] = stub_pipeline["make"](count=7, missing=5)
        result = invoke(tmp_path, "--week", "2026-08-10")
        assert result.exit_code != 0
        assert not (tmp_path / "index.html").exists()

    def test_keeps_the_previous_edition_when_the_gate_fails(self, tmp_path, stub_pipeline):
        invoke(tmp_path, "--week", "2026-08-03")
        stub_pipeline["stories"] = stub_pipeline["make"](count=2)
        invoke(tmp_path, "--week", "2026-08-10")
        assert json.loads((tmp_path / "data" / "latest.json").read_text())["week"] == "2026-08-03"

    def test_skip_gate_publishes_anyway(self, tmp_path, stub_pipeline):
        stub_pipeline["stories"] = stub_pipeline["make"](count=7, missing=5)
        result = invoke(tmp_path, "--week", "2026-08-10", "--skip-gate")
        assert result.exit_code == 0
        assert (tmp_path / "index.html").exists()

    def test_dry_run_writes_nothing(self, tmp_path, stub_pipeline):
        result = invoke(tmp_path, "--week", "2026-08-10", "--dry-run")
        assert result.exit_code == 0
        assert not (tmp_path / "index.html").exists()
        assert not (tmp_path / "data").exists()
        assert "Story 0" in result.stdout

    def test_rebuilds_the_archive_pages_for_every_edition(self, tmp_path, stub_pipeline):
        invoke(tmp_path, "--week", "2026-08-03")
        invoke(tmp_path, "--week", "2026-08-10")
        assert (tmp_path / "archive" / "2026-08-03.html").exists()
        assert (tmp_path / "archive" / "2026-08-10.html").exists()


class TestRunMetrics:
    def metrics_path(self, tmp_path, week="2026-08-10"):
        return tmp_path / "data" / "runs" / f"{week}.json"

    def test_writes_a_run_record(self, tmp_path, stub_pipeline):
        invoke(tmp_path, "--week", "2026-08-10")
        record = json.loads(self.metrics_path(tmp_path).read_text())
        assert record["week"] == "2026-08-10"
        assert "stage_seconds" in record

    def test_no_metrics_skips_the_record(self, tmp_path, stub_pipeline):
        invoke(tmp_path, "--week", "2026-08-10", "--no-metrics")
        assert not self.metrics_path(tmp_path).exists()
        assert (tmp_path / "index.html").exists()

    def test_dry_run_writes_no_record(self, tmp_path, stub_pipeline):
        invoke(tmp_path, "--week", "2026-08-10", "--dry-run")
        assert not self.metrics_path(tmp_path).exists()

    def test_a_failed_metrics_write_does_not_fail_the_run(
        self, tmp_path, stub_pipeline, monkeypatch
    ):
        """Measurement is never allowed to cost us an edition."""

        def explode(self):
            raise OSError("disk full")

        monkeypatch.setattr(main.metrics.RunMetrics, "to_dict", explode)
        result = invoke(tmp_path, "--week", "2026-08-10")
        assert result.exit_code == 0
        assert (tmp_path / "index.html").exists()
        assert not self.metrics_path(tmp_path).exists()

    def test_records_the_run_even_when_the_gate_rejects_the_digest(self, tmp_path, stub_pipeline):
        """A refused week is exactly the week whose numbers you want to read."""
        stub_pipeline["stories"] = stub_pipeline["make"](count=7, missing=5)
        result = invoke(tmp_path, "--week", "2026-08-10")
        assert result.exit_code != 0
        assert self.metrics_path(tmp_path).exists()


class TestAlreadyPublished:
    """The 2026-08-31 edition was built twice — a manual dispatch, then the
    delayed cron — and the second run silently replaced the first. Rebuilding
    a published week must be an explicit decision."""

    def test_skips_a_week_that_is_already_published(self, tmp_path, stub_pipeline):
        archive = tmp_path / "data" / "archive"
        archive.mkdir(parents=True)
        (archive / "2026-08-10.json").write_text('{"week": "2026-08-10", "stories": []}')
        before = (archive / "2026-08-10.json").read_text()

        result = invoke(tmp_path, "--week", "2026-08-10")
        assert result.exit_code == 0
        assert "already published" in result.output
        assert (archive / "2026-08-10.json").read_text() == before
        assert not (tmp_path / "index.html").exists()

    def test_force_rebuilds_a_published_week(self, tmp_path, stub_pipeline):
        archive = tmp_path / "data" / "archive"
        archive.mkdir(parents=True)
        (archive / "2026-08-10.json").write_text('{"week": "2026-08-10", "stories": []}')

        result = invoke(tmp_path, "--week", "2026-08-10", "--force")
        assert result.exit_code == 0
        assert (tmp_path / "index.html").exists()

    def test_dry_run_still_runs_on_a_published_week(self, tmp_path, stub_pipeline):
        archive = tmp_path / "data" / "archive"
        archive.mkdir(parents=True)
        (archive / "2026-08-10.json").write_text('{"week": "2026-08-10", "stories": []}')

        result = invoke(tmp_path, "--week", "2026-08-10", "--dry-run")
        assert result.exit_code == 0
        assert '"week"' in result.output


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
