"""Ask the live Space and time it from here: either dev questions for latency and cost, or a
smoke check after a deploy.

uv run python eval/run_live.py 30         30 dev questions, saved to eval/results/*_live.json
uv run python eval/run_live.py --smoke    three known questions; fails unless each comes back right

The Space allows 10 questions per visitor in 10 minutes, so a longer run pauses after every 10.
A question asked before comes from the Space's cache, so restart the Space first for fresh times.
"""

import argparse
import json
import random
import re
import subprocess
import time
from datetime import date
from pathlib import Path

from gradio_client import Client
from huggingface_hub import get_token

from lotse.config import ROOT, SPACE_REPO
from lotse.generation.answer import NO_ANSWER

HERE = Path(__file__).parent
# Two that the Space answers with a source, one it must refuse. The first two are on its page.
SMOKE = [
    ("Muss auch ein Kleinbetrieb die Gefährdungsbeurteilung für Gefahrstoffe aufschreiben?", True),
    ("Wie lange darf ich am Tag höchstens arbeiten?", True),
    ("Wie viele Urlaubstage stehen mir im Jahr zu?", False),
]


def status(page: str) -> str:
    """What the page shows: an answer with sources, a refusal, or anything else."""
    if NO_ANSWER in page:
        return "no_answer"
    return "answered" if "<blockquote" in page else "other"


def ask(client: Client, question: str) -> tuple[str, float, str | None]:
    start = time.perf_counter()
    try:
        page, error = client.predict(question, api_name="/ask"), None
    except Exception as exception:  # noqa: BLE001 - recorded with the row
        page, error = "", f"{type(exception).__name__}: {exception}"[:300]
    return page, round(time.perf_counter() - start, 2), error


def smoke(client: Client) -> None:
    failed = []
    for question, answerable in SMOKE:
        page, seconds, error = ask(client, question)
        shown = status(page)
        print(f"{seconds:6.2f} s  {shown:9}  {question}")
        if error or shown != ("answered" if answerable else "no_answer"):
            failed.append(error or " ".join(re.sub(r"<[^>]+>", " ", page).split())[:200])
    if failed:
        raise SystemExit("Smoke check failed:\n" + "\n".join(failed))


def measure(client: Client, count: int) -> None:
    lines = (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    questions = [q for q in map(json.loads, lines) if q["split"] == "dev"]
    random.Random(7).shuffle(questions)  # the same 30 as earlier runs
    rows = []
    for number, question in enumerate(questions[:count]):
        if number and number % 10 == 0:
            print("waiting out the rate limit")
            time.sleep(610)
        page, seconds, error = ask(client, question["question"])
        rows.append({"id": question["id"], "type": question["type"], "seconds": seconds,
                     "status": status(page), "error": error})  # fmt: skip
        print(rows[-1])
    git = ["git", "describe", "--always", "--dirty", "--abbrev=7", "--exclude=*"]
    commit = subprocess.run(git, cwd=ROOT, capture_output=True, text=True, check=False)
    result = {
        "date": date.today().isoformat(),
        "space": SPACE_REPO,
        "commit": commit.stdout.strip() or "uncommitted",
        "split": "dev",
        "note": "seconds per row are timed by this client, network and queue included; "
        "metrics are the Space's /metrics after the questions",
        "metrics": client.predict(api_name="/metrics"),
        "rows": rows,
    }
    path = HERE / "results" / f"{result['date']}_{result['commit']}_live.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2), f"\nsaved to {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count", nargs="?", type=int, default=30, help="dev questions to ask")
    parser.add_argument("--smoke", action="store_true", help="the three known questions only")
    args = parser.parse_args()
    # A token gives the Space's GPU quota of its owner instead of an anonymous visitor's.
    client = Client(SPACE_REPO, token=get_token(), verbose=False)
    if args.smoke:
        smoke(client)
    else:
        measure(client, args.count)


if __name__ == "__main__":
    main()
