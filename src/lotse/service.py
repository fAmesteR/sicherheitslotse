"""The JSON API around the demo: /ask, /health and /metrics.

app/server.py adds the page and starts it. Nothing here loads a model, so the tests can run it.
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from lotse.demo import FAILED, MAX_QUESTION_CHARS, OVER_BUDGET, TOO_MANY, Demo, visitor


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


def create_app(demo: Demo) -> FastAPI:
    app = FastAPI(title="Sicherheitslotse", version="0.1.0")

    @app.post("/ask")
    def ask(body: Question, request: Request) -> dict:
        """One question in, the checked answer with its sources out."""
        host = request.client.host if request.client else None
        who = visitor(host, request.headers.get("x-forwarded-for", ""))
        outcome = demo.answer(body.question, who)
        if isinstance(outcome, str):  # a notice: the question never produced a result
            codes = {TOO_MANY: 429, OVER_BUDGET: 503, FAILED: 503}
            raise HTTPException(codes.get(outcome, 422), outcome or "Die Frage ist leer.")
        return {
            "status": outcome.status,
            "answer": outcome.text,
            "citations": [
                {
                    "source": citation.source,
                    "quote": citation.quote,
                    **{
                        key: outcome.passages[citation.source - 1][key]
                        for key in ("doc", "section", "heading", "url", "version_date")
                    },
                }
                for citation in outcome.citations
            ],
        }

    @app.get("/health")
    async def health() -> dict:
        """Answers only once the index, the key and both models are loaded. Async, so it runs on
        the event loop and not in the thread pool, where slow questions could make it wait."""
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> dict:
        """Latency and cost of the queries since the start."""
        return demo.summary()

    return app
