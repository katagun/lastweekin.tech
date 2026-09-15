"""Verify that every model named in config.yaml still exists on OpenRouter.

Hosted model IDs are retired without notice. This project shipped a chain of
four models of which three had already been withdrawn, leaving the pipeline one
outage away from publishing nothing. Failing loudly on a schedule turns that
into a Tuesday notification instead of a Monday of empty summaries.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests

from lastweekintech.config import get_config

CATALOGUE_URL = "https://openrouter.ai/api/v1/models"
TIMEOUT = 30


def available_model_ids(url: str = CATALOGUE_URL) -> set[str]:
    """Every model id the provider currently offers."""
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return {model["id"] for model in response.json()["data"]}


def main() -> int:
    config = get_config()
    summarizer = config.summarizer

    briefing = config.ai_daily.briefing
    # Hugging Face models are routed elsewhere and are not in this catalogue.
    hosted = [
        summarizer.model_name,
        *summarizer.fallback_models,
        briefing.model_name,
        *briefing.fallback_models,
    ]
    checked = [m for m in hosted if m not in set(summarizer.huggingface_models)]

    try:
        available = available_model_ids()
    except (requests.RequestException, OSError) as exc:
        print(f"Could not reach the OpenRouter catalogue: {exc}", file=sys.stderr)
        return 2

    missing = [model for model in checked if model not in available]
    for model in checked:
        print(f"{'MISSING' if model in missing else 'ok     '}  {model}")

    if missing:
        print(
            f"\n{len(missing)} of {len(checked)} configured models no longer exist. "
            "Update summarizer.model_name / fallback_models in config.yaml.",
            file=sys.stderr,
        )
        return 1

    if checked and checked[0] not in available:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
