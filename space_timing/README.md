---
title: Rerank Timing
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12.12"
app_file: app.py
preload_from_hub:
  - BAAI/bge-reranker-v2-m3
---

Week 1 test for Sicherheitslotse: how long does `bge-reranker-v2-m3` take on ZeroGPU, queue
included? It produced the ZeroGPU row of the reranker timing in the main README. The Space it ran on
was later renamed and reused for the demo, so this folder is kept only as the record of how that
number was measured. `pairs.json` comes from `scripts/time_rerank.py --export` and holds law text
only.

To run it again, create a Space from this folder and select ZeroGPU in its settings. The README
cannot set it.
