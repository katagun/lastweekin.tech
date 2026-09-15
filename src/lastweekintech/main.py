"""
Main CLI application for the LastWeekIn.Tech pipeline.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from lastweekintech import ai_daily, metrics, pipeline
from lastweekintech.briefer import Briefer
from lastweekintech.config import ConfigError, get_config
from lastweekintech.editor import Editor
from lastweekintech.summarizer import Summarizer
from lastweekintech.validation import DigestValidationError, assert_publishable

app = typer.Typer(add_completion=False)


@app.command("run")
def run(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", "-d", help="Where latest.json and the archive are written."),
    ] = Path("data"),
    site_dir: Annotated[
        Path,
        typer.Option("--site-dir", "-s", help="Where the static site is written."),
        # A directory of its own, not the repository root: the host serves this
        # tree verbatim, and the root holds source, tests and tooling that have
        # no business being fetchable from the public site.
    ] = Path("public"),
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config.yaml."),
    ] = None,
    week: Annotated[
        str | None,
        typer.Option("--week", "-w", help="Edition date (YYYY-MM-DD). Defaults to today, UTC."),
    ] = None,
    skip_gate: Annotated[
        bool,
        typer.Option("--skip-gate", help="Publish even if the digest fails validation."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild and overwrite an already-published week."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the edition instead of writing any files."),
    ] = False,
    no_metrics: Annotated[
        bool,
        typer.Option("--no-metrics", help="Skip writing the run record to data/runs/."),
    ] = False,
):
    """Run the LastWeekIn.Tech data pipeline."""
    now = datetime.now(UTC)
    week = week or now.strftime("%Y-%m-%d")

    # A delayed cron firing after a manual dispatch (or the reverse) must not
    # silently replace an edition readers already saw. Rebuilding a published
    # week is an explicit decision, not a race outcome.
    already_published = data_dir / "archive" / f"{week}.json"
    if already_published.exists() and not force and not dry_run:
        typer.secho(
            f"The {week} edition is already published ({already_published}); "
            "use --force to rebuild it.",
            fg=typer.colors.YELLOW,
        )
        return

    try:
        config = get_config(config_path)
        summarizer = Summarizer(config.summarizer)
        editor = Editor(config.editor) if config.editor.enabled else None
    except (ConfigError, ValueError) as e:
        typer.secho(f"Configuration error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e

    run_metrics = None if no_metrics else metrics.RunMetrics(week=week)
    digest = pipeline.build_digest(
        config,
        summarizer,
        now=now,
        editor=editor,
        # Read before the run so a story still riding a week-old Hacker News
        # thread does not top two editions in a row.
        editions=pipeline.list_editions(data_dir),
        metrics=run_metrics,
    )
    stories = digest.stories

    def save_metrics() -> None:
        if run_metrics is not None and not dry_run:
            metrics.write_run_metrics(run_metrics, data_dir)

    # The gate runs before anything is written, so a bad week leaves last
    # week's edition live rather than replacing it with something broken. The
    # run record is kept either way: a refused week is exactly the week whose
    # numbers you want to read afterwards.
    try:
        assert_publishable(stories, config)
    except DigestValidationError as e:
        if not skip_gate:
            save_metrics()
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e
        logging.warning(f"--skip-gate set; publishing anyway. {e}")

    edition = pipeline.build_edition(stories, week=week, now=now, intro=digest.intro)

    if dry_run:
        typer.echo(json.dumps(edition, indent=2, ensure_ascii=False))
        return

    pipeline.save_edition(edition, data_dir)
    pipeline.generate_site(
        pipeline.list_editions(data_dir),
        output_dir=site_dir,
        site_url=config.site_url,
    )
    save_metrics()
    typer.secho(f"Published the edition for {week}.", fg=typer.colors.GREEN)


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


if __name__ == "__main__":
    app()
