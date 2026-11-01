"""Throwaway week 1 test: rerank timing on ZeroGPU, queue included. Delete after the decision."""

import json
import os
import statistics
import time
from pathlib import Path

import gradio as gr
import spaces  # ZeroGPU needs this imported before torch sets up CUDA
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "BAAI/bge-reranker-v2-m3"
DEVICE = os.environ.get("TIMING_DEVICE", "cuda")  # TIMING_DEVICE=cpu for a local dry run
SETS = json.loads(Path(__file__).with_name("pairs.json").read_text(encoding="utf-8"))

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(DEVICE).eval()


@spaces.GPU(duration=15)
def rerank(question: str, passages: list[str]) -> tuple[float, str]:
    start = time.perf_counter()
    batch = tokenizer(
        [[question, passage] for passage in passages],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.inference_mode():
        model(**batch).logits.view(-1).tolist()  # tolist waits for the GPU to finish
    device = torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu"
    return time.perf_counter() - start, device


def timed(question: str, passages: list[str]) -> tuple[float, float, str]:
    """Seconds as a visitor feels them, seconds inside the GPU function, and the device."""
    start = time.perf_counter()
    inside, device = rerank(question, passages)
    return time.perf_counter() - start, inside, device


def summary(runs: list[tuple[float, float, str]]) -> dict:
    return {
        "calls": len(runs),
        "total_s_p50": round(statistics.median(run[0] for run in runs), 2),
        "total_s_max": round(max(run[0] for run in runs), 2),
        "inside_s_p50": round(statistics.median(run[1] for run in runs), 2),
    }


def measure() -> dict:
    questions = [(item["question"], item["passages"]) for item in SETS["typical"]]
    first = timed(*questions[0])
    typical = [timed(question, passages) for question, passages in questions]
    long = [timed(question, SETS["long"]) for question, _ in questions]
    return {
        "device": first[2],
        "candidates": len(SETS["long"]),
        "first_call_total_s": round(first[0], 2),
        "typical": summary(typical),
        "long": summary(long),
    }


demo = gr.Interface(
    fn=measure,
    inputs=None,
    outputs=gr.JSON(label="Result"),
    title="Rerank timing",
    description="Week 1 test for Sicherheitslotse. Run it a few times, also after the Space slept.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
