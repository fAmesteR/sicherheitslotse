---
title: Sicherheitslotse
emoji: 🦺
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12.12"
app_file: app/app.py
short_description: Arbeitsschutzrecht mit geprüften Zitaten
preload_from_hub:
  - BAAI/bge-m3 config.json,pytorch_model.bin,sentencepiece.bpe.model,special_tokens_map.json,tokenizer.json,tokenizer_config.json
  - BAAI/bge-reranker-v2-m3 config.json,model.safetensors,sentencepiece.bpe.model,special_tokens_map.json,tokenizer.json,tokenizer_config.json
---

Fragen zum deutschen Arbeitsschutzrecht, beantwortet mit wörtlichen Zitaten aus 11 Gesetzen und
Verordnungen. Ein Programm prüft jedes Zitat und jede Zahl gegen den Gesetzestext, bevor eine
Antwort angezeigt wird.

Code, Messwerte und Grenzen: https://github.com/fAmesteR/sicherheitslotse

This file is deployed as the Space's README by `scripts/deploy_space.py`. The Space needs ZeroGPU
hardware, which is chosen in its settings, and two secrets: `MISTRAL_API_KEY`, and a read-only
`HF_TOKEN` for the private index dataset.
