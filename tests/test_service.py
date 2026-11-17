import json
import logging
import re

from fastapi.testclient import TestClient

from lotse import demo as demo_module
from lotse.config import ANSWER_MODEL, ROOT
from lotse.demo import MAX_QUESTION_CHARS, Demo, RateLimit
from lotse.generation import llm
from lotse.generation.answer import Citation
from lotse.metrics import percentile
from lotse.pipeline import Result
from lotse.service import create_app

LAW = {
    "doc": "ArbZG",
    "section": "§ 4",
    "heading": "Ruhepausen",
    "text": "Die Arbeit ist durch Ruhepausen zu unterbrechen.",
    "url": "https://example.org/arbzg/__4.html",
    "version_date": "2025-12-22",
}
QUOTE = Citation(source=1, quote="durch Ruhepausen zu unterbrechen")
ANSWERED = Result("answered", "Pausen sind Pflicht [1].", [QUOTE], [LAW])


def service(limit: int = 10, seconds: float = 2.0):
    """The API over a stand-in pipeline. Each question takes `seconds` on a fake clock and uses
    1,000 prompt tokens and 100 completion tokens, 0.00021 USD at the answer model's prices."""
    state = {"now": 0.0}

    def ask(question: str) -> Result:
        if "kaputt" in question:
            raise ConnectionError("no network")
        state["now"] += seconds
        llm.charge(ANSWER_MODEL, 1000, 100)
        return ANSWERED

    demo = Demo(ask, RateLimit(limit), clock=lambda: state["now"])
    return TestClient(create_app(demo))


def test_an_answer_comes_back_with_its_sources():
    client = service()
    reply = client.post("/ask", json={"question": "Muss ich Pause machen?"})
    assert reply.status_code == 200
    assert reply.json() == {
        "status": "answered",
        "answer": "Pausen sind Pflicht [1].",
        "citations": [
            {
                "source": 1,
                "quote": "durch Ruhepausen zu unterbrechen",
                "doc": "ArbZG",
                "section": "§ 4",
                "heading": "Ruhepausen",
                "url": "https://example.org/arbzg/__4.html",
                "version_date": "2025-12-22",
            }
        ],
    }


def test_bad_requests_get_the_right_status_code():
    client = service(limit=1)
    assert client.post("/ask", json={"question": "x" * (MAX_QUESTION_CHARS + 1)}).status_code == 422
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "Ist das kaputt?"}).status_code == 503
    assert client.post("/ask", json={"question": "Zweite Frage?"}).status_code == 429


def test_every_query_is_logged_with_latency_and_cost_but_without_the_question(caplog):
    client = service()
    with caplog.at_level(logging.INFO, logger="lotse.queries"):
        client.post("/ask", json={"question": "Muss ich Pause machen?"})
        client.post("/ask", json={"question": "muss ich pause machen?"})
    fresh, cached = (json.loads(record.message) for record in caplog.records)
    assert (fresh["status"], fresh["cached"], fresh["seconds"]) == ("answered", False, 2.0)
    assert (fresh["prompt_tokens"], fresh["completion_tokens"], fresh["usd"]) == (
        1000,
        100,
        0.00021,
    )
    assert (cached["cached"], cached["seconds"], cached["usd"]) == (True, 0.0, 0.0)
    assert "Pause" not in caplog.text and "testclient" not in caplog.text


def test_metrics_report_latency_and_cost_per_thousand_queries():
    client = service()
    for number in range(4):
        client.post("/ask", json={"question": f"Frage {number}?"})
    client.post("/ask", json={"question": "Frage 0?"})  # from the cache
    client.post("/ask", json={"question": "Ist das kaputt?"})
    metrics = client.get("/metrics").json()
    assert metrics["by_status"] == {"answered": 5, "failed": 1}
    assert (metrics["from_cache"], metrics["p50_seconds"], metrics["p95_seconds"]) == (1, 2.0, 2.0)
    assert metrics["usd_per_1000_queries"] == 0.168  # 4 paid answers over 5 served queries
    assert client.get("/health").json() == {"status": "ok"}


def test_the_api_limits_by_the_same_visitor_as_the_page(monkeypatch):
    # Behind a trusted proxy each visitor has their own limit, as on the page.
    monkeypatch.setattr(demo_module, "TRUST_PROXY", True)
    client = service(limit=1)
    codes = [
        client.post(
            "/ask",
            json={"question": f"Frage {n}?"},
            headers={"X-Forwarded-For": f"93.184.216.{n}"},
        ).status_code
        for n in range(3)
    ]
    assert codes == [200, 200, 200]
    # Without one the header is ignored, so it can't buy more questions.
    monkeypatch.setattr(demo_module, "TRUST_PROXY", False)
    client = service(limit=1)
    codes = [
        client.post(
            "/ask", json={"question": f"Frage {n}?"}, headers={"X-Forwarded-For": f"1.2.3.{n}"}
        ).status_code
        for n in range(3)
    ]
    assert codes == [200, 429, 429]


def test_a_percentile_is_the_nearest_rank():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == 10
    assert percentile([1, 2, 3, 4], 0.50) == 2
    assert percentile([], 0.95) is None


def test_the_image_installs_the_locked_torch_version():
    """The Dockerfile names its CPU build of torch by hand, so a new lock must update it too."""
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    locked = re.search(r'^name = "torch"\nversion = "([^"]+)"', lock, re.MULTILINE)[1]
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.findall(r"torch==([\w.]+)\+cpu --hash=sha256:[0-9a-f]{64}", dockerfile) == [locked]


def test_ci_uses_the_same_uv_as_the_image():
    """A newer uv may export the same lock differently, so every workflow pins the image's uv."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    image = re.search(r"astral-sh/uv:([\d.]+)@", dockerfile)[1]
    for workflow in (ROOT / ".github" / "workflows").glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        steps = text.count("astral-sh/setup-uv@")
        assert text.count(f'version: "{image}"') == steps, workflow.name
