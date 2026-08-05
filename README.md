# LLM Persona Effects in FOMC Rate-Setting

This repository reproduces the main experiment and the weight-generation robustness check for the FOMC-style monetary policy decision task. The project compares DeepSeek and Qwen under policy-persona, no-persona, and placebo conditions, then reports the downstream robustness results.

## Team
Team 2 — AI in Econ Online PBL, 2026.

## Setup

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY="your_key_here"
export DASHSCOPE_API_KEY="your_key_here"
```

## Reproduce the headline figure

```bash
python main_experiment.py
python weight_generation.py
```

The main analysis script produces the headline experimental outputs. The weight-generation script produces the auxiliary policy-weight robustness experiment.

## Repository structure

```text
README.md
requirements.txt
LICENSE
main_experiment.py
weight_generation.py
prompts/
results/
```

## Models

- DeepSeek: `deepseek-v4-flash`
- Qwen: `qwen-plus`
- The experiment uses the same prompt structures and repeated runs documented in the prompt text files in `prompts/`.

## Notes

- Do not commit any API keys.
- Store your own keys only in the local environment.
- The parsed result tables in `results/` are the public-facing artifacts for replication.
