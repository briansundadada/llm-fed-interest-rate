"""Generate explicit policy-goal weights only (no interest-rate decision).

Design:
    4 persona conditions × 3 scenarios × 15 repeats = 180 observations.
    "none" keeps the common FOMC role but adds no policy-persona guidance.

Outputs:
    q1_weights_180.csv          run-level weights
    q1_weights_summary.csv      cell-level means and standard deviations

The API key is read from DEEPSEEK_KEY.txt. It is never printed or exported.
The output is checkpointed after every valid API response and can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


MODEL = "deepseek-v4-flash"
API_URL = "https://api.deepseek.com/v1/chat/completions"
N_REPEATS = 15

PERSONAS = {
    "hawk": (
        "You place greater weight on price stability and returning inflation "
        "to the 2% target, while still recognizing maximum employment."
    ),
    "centrist": (
        "You weigh price stability and maximum employment roughly equally, "
        "while also considering output stabilization."
    ),
    "dove": (
        "You place greater weight on maximum employment and labor-market "
        "conditions, while still recognizing price stability."
    ),
    "none": None,
}

PERSONA_DISPLAY = {
    "hawk": "Hawk",
    "centrist": "Centrist",
    "dove": "Dove",
    "none": "No Persona",
}

SCENARIOS = {
    "sc1": {
        "scenario_label": "Scenario A",
        "fed_funds": 5.00,
        "pce_inflation_yoy": 3.8,
        "unemployment": 3.8,
        "inflation_exp_5y": 2.7,
        "real_gdp": 101.0,
        "potential_gdp": 100.0,
    },
    "sc2": {
        "scenario_label": "Scenario B",
        "fed_funds": 5.00,
        "pce_inflation_yoy": 2.1,
        "unemployment": 6.0,
        "inflation_exp_5y": 2.1,
        "real_gdp": 98.5,
        "potential_gdp": 100.0,
    },
    "sc3": {
        "scenario_label": "Scenario C",
        "fed_funds": 5.00,
        "pce_inflation_yoy": 3.1,
        "unemployment": 4.9,
        "inflation_exp_5y": 2.4,
        "real_gdp": 99.5,
        "potential_gdp": 100.0,
    },
}

RAW_FIELDS = [
    "persona", "scenario_id", "scenario_label", "run_id", "model",
    "inflation_weight", "employment_weight", "growth_weight",
    "weights_sum", "weight_difference", "total_tokens",
]

SUMMARY_FIELDS = [
    "Scenario", "Persona", "Inflation Weight", "Employment Weight",
    "Growth Weight", "Weight Difference",
]


def load_key(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig").strip()
    if "=" in text:
        name, value = text.split("=", 1)
        if name.strip() in {"DEEPSEEK_KEY", "DEEPSEEK_API_KEY"}:
            text = value.strip().strip("'\"")
    if not text:
        raise ValueError("The API key file is empty.")
    return text


def system_prompt(persona: str) -> str:
    if persona == "none":
        # Match the original no_persona condition exactly: keep the common
        # FOMC role and task instructions, but inject no persona-related text.
        persona_block = ""
    else:
        persona_block = (
            f"Your policy persona is {persona}.\n{PERSONAS[persona]}"
        )
    return f"""
You are a senior FOMC policymaker.
{persona_block}

Your only task is to assign prospective decision weights. Do not recommend,
imply, or discuss any interest-rate action.

Allocate weights to:
1. inflation stabilization,
2. employment support,
3. growth/output stabilization.

Each weight must be between 0 and 1. The three weights must sum to exactly
1.00, with a tolerance of 0.01.

Return JSON only:
{{
  "inflation_weight": <number>,
  "employment_weight": <number>,
  "growth_weight": <number>
}}
""".strip()


def user_prompt(scenario: dict) -> str:
    return f"""
QUARTERLY MACRO BRIEFING ({scenario['scenario_label']}):
- Current effective fed funds rate: {scenario['fed_funds']:.2f}%
- Unemployment rate: {scenario['unemployment']:.1f}%
- PCE inflation (YoY): {scenario['pce_inflation_yoy']:.1f}%
- 5-year inflation expectations: {scenario['inflation_exp_5y']:.1f}%
- Real GDP index: {scenario['real_gdp']:.1f}
- Potential GDP index: {scenario['potential_gdp']:.1f}

Assign the three policy-goal weights only. Do not discuss a rate decision.
""".strip()


def call_api(key: str, persona: str, scenario: dict, retries: int = 4):
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt(persona)},
            {"role": "user", "content": user_prompt(scenario)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.5,
    }).encode("utf-8")
    for attempt in range(retries):
        request = urllib.request.Request(
            API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                envelope = json.loads(response.read().decode("utf-8"))
            parsed = json.loads(envelope["choices"][0]["message"]["content"])
            return parsed, envelope.get("usage", {}).get("total_tokens")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise RuntimeError(f"API HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                KeyError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"API/JSON failure: {exc}") from exc
        time.sleep((2 ** attempt) + random.random())
    raise RuntimeError("Unreachable retry state.")


def validate(parsed: dict) -> dict:
    fields = ["inflation_weight", "employment_weight", "growth_weight"]
    if any(field not in parsed for field in fields):
        raise ValueError("Missing weight field.")
    values = [float(parsed[field]) for field in fields]
    if any(value < 0 or value > 1 for value in values):
        raise ValueError(f"Weight outside [0,1]: {values}")
    total = sum(values)
    if abs(total - 1) > 0.011:
        raise ValueError(f"Weights sum to {total}, not 1.")
    return {
        "inflation_weight": values[0],
        "employment_weight": values[1],
        "growth_weight": values[2],
        "weights_sum": total,
        "weight_difference": values[0] - values[1],
    }


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            (row["persona"], row["scenario_id"], int(row["run_id"]))
            for row in csv.DictReader(handle)
        }


def append_row(path: Path, row: dict) -> None:
    fresh = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values)
            / (len(values) - 1)) ** 0.5


def write_summary(raw_path: Path, summary_path: Path) -> None:
    with raw_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["persona"], row["scenario_id"]), []).append(row)
    output = []
    for scenario_id, scenario in SCENARIOS.items():
        for persona in PERSONAS:
            group = groups.get((persona, scenario_id), [])
            if not group:
                continue
            averages = {}
            for field in [
                "inflation_weight", "employment_weight",
                "growth_weight", "weight_difference",
            ]:
                values = [float(row[field]) for row in group]
                averages[field] = mean(values)
            output.append({
                "Scenario": scenario["scenario_label"].removeprefix("Scenario "),
                "Persona": PERSONA_DISPLAY[persona],
                "Inflation Weight": f"{averages['inflation_weight']:.1%}",
                "Employment Weight": f"{averages['employment_weight']:.1%}",
                "Growth Weight": f"{averages['growth_weight']:.1%}",
                "Weight Difference": f"{averages['weight_difference']:+.1%}",
            })
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(output)


def collect(args) -> None:
    key = load_key(args.key_file)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(args.raw_output)
    tasks = [
        (persona, scenario_id, scenario, run_id)
        for persona in PERSONAS
        for scenario_id, scenario in SCENARIOS.items()
        for run_id in range(1, args.repeats + 1)
    ]
    if args.smoke_test:
        tasks = [("hawk", "sc3", SCENARIOS["sc3"], 0)]
        done = set()
    for index, (persona, scenario_id, scenario, run_id) in enumerate(tasks, 1):
        if (persona, scenario_id, run_id) in done:
            continue
        print(f"[{index}/{len(tasks)}] {persona}/{scenario_id}/run={run_id}",
              flush=True)
        last_error = None
        for attempt in range(3):
            try:
                parsed, tokens = call_api(key, persona, scenario)
                weights = validate(parsed)
                row = {
                    "persona": persona,
                    "scenario_id": scenario_id,
                    "scenario_label": scenario["scenario_label"],
                    "run_id": run_id,
                    "model": MODEL,
                    **weights,
                    "total_tokens": tokens,
                }
                if args.smoke_test:
                    print(json.dumps(row, ensure_ascii=False, indent=2))
                else:
                    append_row(args.raw_output, row)
                last_error = None
                break
            except (ValueError, RuntimeError) as exc:
                last_error = exc
                print(f"  retry {attempt + 1}/3: {exc}", flush=True)
        if last_error:
            raise last_error
    if not args.smoke_test:
        write_summary(args.raw_output, args.summary_output)
        print(f"Complete: {args.raw_output}")
        print(f"Summary:  {args.summary_output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-file", type=Path, default=Path("DEEPSEEK_KEY.txt"))
    parser.add_argument("--raw-output", type=Path,
                        default=Path("q1_weights_180.csv"))
    parser.add_argument("--summary-output", type=Path,
                        default=Path("q1_weights_summary.csv"))
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--smoke-test", action="store_true")
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
