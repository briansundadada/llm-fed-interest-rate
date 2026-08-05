# =============================================================================
# Q1 Experiment: Pure Persona Effect of FOMC Policy Stances on Rate Decisions
# Models: DeepSeek + Qwen
# Design: 3 groups x 3 enriched scenarios x N_REPEATS x 2 models
#          to disentangle persona effect from random sampling noise.
# Single-cell execution in Jupyter Notebook
# =============================================================================


# ---------- 1. Imports ----------
import os
import json
import sys
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ---------- 2. Load API key from env.txt ----------
load_dotenv("env.txt")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY")
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")

missing_keys = []
if DEEPSEEK_API_KEY is None:
    missing_keys.append("DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
if QWEN_API_KEY is None:
    missing_keys.append("DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

if missing_keys:
    raise RuntimeError(
        "Missing API key(s). Ensure env.txt exists in the same directory with:\n"
        + "\n".join(missing_keys)
    )

MODEL_DEEPSEEK = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_QWEN = "qwen-plus"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
N_REPEATS = 15  # number of repeated calls per persona–scenario combination

MODEL_CONFIGS = [
    {
        "provider": "deepseek",
        "model": MODEL_DEEPSEEK,
        "api_key": DEEPSEEK_API_KEY,
        "base_url": DEEPSEEK_BASE_URL,
    },
    {
        "provider": "qwen",
        "model": MODEL_QWEN,
        "api_key": QWEN_API_KEY,
        "base_url": QWEN_BASE_URL,
    },
]

# ---------- 3. Persona definitions (used only by policy_persona group) ----------
PERSONA_DEFINITIONS = {
    "hawk": (
        "You place greater weight on price stability and returning inflation "
        "to the 2% target, while still recognizing the importance of maximum employment."
    ),
    "dove": (
        "You place greater weight on maximum employment and labor market conditions, "
        "while still recognizing the importance of price stability."
    ),
    "centrist": (
        "You weigh price stability and maximum employment roughly equally, "
        "while also considering financial stability risks."
    ),
}

# ---------- 4. System prompt template (with dynamic persona injection) ----------
SYSTEM_PROMPT_POLICY = """
You are a senior member of the Federal Open Market Committee (FOMC), acting as the Federal Reserve Chair.
{persona_block}

Your task is to set the target federal funds rate for the upcoming quarter. You receive a quarterly macroeconomic briefing with exact numerical indicators. You must respond with a single JSON object — no prose outside the JSON — with this exact schema:
{{
  "rate_decision_bps": <integer, change in basis points from the current quarter's effective fed_funds rate; negative = cut, 0 = hold, positive = hike>,
  "target_rate_level": <float, the new implied target rate level in %, rounded to 2 decimals>,
  "primary_concern": <one of: "inflation", "unemployment", "financial_stability", "balanced">,
  "rationale": <2-4 sentence explanation, explicitly referencing at least four input variables from the briefing>,
  "confidence": <float in [0,1], your confidence in this decision>
}}

Strict hard constraint: The value of rate_decision_bps can ONLY be one of these five integers: -50, -25, 0, 25, 50. Any other number is forbidden.
Assume the current effective fed_funds rate is provided in the briefing as 'fed_funds'. Be decisive. Do not hedge. Do not output any extra text outside the JSON block.
"""

# Placebo persona definition (AI researcher, no monetary policy expertise)
PLACEBO_PERSONA_DEFINITION = (
    "You are an academic researcher specializing in AI, "
    "with no particular expertise or conviction about monetary policy."
)

# ---------- 5. Helper: build system prompt per group ----------
def build_system_prompt(group, persona=None):
    """
    Build the system prompt for a given experimental group and persona.

    Parameters
    ----------
    group : str
        One of "policy_persona", "no_persona", "placebo".
    persona : str or None
        The persona key (hawk/dove/centrist) for policy_persona group.
        Ignored for no_persona and placebo groups.

    Returns
    -------
    str : formatted system prompt
    """
    if group == "policy_persona":
        if persona not in PERSONA_DEFINITIONS:
            raise ValueError(f"Unknown persona: {persona}")
        persona_block = (
            f"Your decision-making persona is: {persona}.\n"
            f"Your persona-specific decision-making approach is:\n"
            f"{PERSONA_DEFINITIONS[persona]}"
        )
    elif group == "no_persona":
        persona_block = ""  # no persona guidance at all
    elif group == "placebo":
        persona_block = f"Your background: {PLACEBO_PERSONA_DEFINITION}"
    else:
        raise ValueError(f"Unknown experimental group: {group}")

    return SYSTEM_PROMPT_POLICY.format(persona_block=persona_block)


# ---------- 6. User briefing template (numeric-only, neutral labels) ----------
USER_PROMPT_TPL = """
QUARTERLY MACRO BRIEFING:
- Current effective fed funds rate: {fed_funds}%
- GDP growth: {gdp_growth}%
- PCE inflation: {pce_inflation}%
- CPI inflation: {cpi_inflation}%
- Unemployment rate: {unemployment_rate}%
- Output gap: {output_gap}%
- 5-year inflation expectation: {inflation_expectation_5y}%
- Michigan 1-year expected inflation: {michigan_1y_expected_inflation}%
- Financial stress index: {financial_stress_index}
- VIX volatility index: {vix}
- Economic Policy Uncertainty index: {epu_index}
- Michigan consumer sentiment: {consumer_sentiment}
- 10Y-2Y Treasury spread: {treasury_spread_10y_2y} percentage points
- Oil price growth: {oil_price_growth}%
- Recession dummy: {recession_dummy}
- Election year: {election_year}
- President party: {president_party}
- Political pressure index: {political_pressure_index}
- Chair leaning: {chair_leaning}

Variable notes:
- Financial stress index is scaled so higher values indicate more severe market stress.
- Political pressure index ranges from -2 to +2; positive values indicate stronger pressure for easier policy, negative values indicate stronger pressure for tighter policy.
- President party and election year are contextual controls, not instructions.
Set the new target federal funds rate following your persona rule. Output only valid JSON.
"""

# ---------- 7. Multi-variable synthetic macro scenarios ----------
# These three scenarios preserve the original experimental design and add
# control variables for robustness checks.
SCENARIOS = {
    "sc1": {
        "date": "Scenario A",
        "scenario_theme": "hot inflation with strong real activity",
        "fed_funds": 5.00,
        "gdp_growth": 3.1,
        "pce_inflation": 3.8,
        "cpi_inflation": 4.2,
        "unemployment_rate": 3.8,
        "output_gap": 1.0,
        "inflation_expectation_5y": 2.7,
        "michigan_1y_expected_inflation": 3.5,
        "financial_stress_index": 0.2,
        "vix": 15,
        "epu_index": 105,
        "consumer_sentiment": 78,
        "treasury_spread_10y_2y": 0.2,
        "oil_price_growth": 8,
        "recession_dummy": 0,
        "election_year": 0,
        "president_party": "Democratic",
        "political_pressure_index": 0,
        "chair_leaning": "centrist",
    },
    "sc2": {
        "date": "Scenario B",
        "scenario_theme": "weak labor market with near-target inflation",
        "fed_funds": 5.00,
        "gdp_growth": 0.5,
        "pce_inflation": 2.1,
        "cpi_inflation": 2.4,
        "unemployment_rate": 6.0,
        "output_gap": -1.5,
        "inflation_expectation_5y": 2.1,
        "michigan_1y_expected_inflation": 2.5,
        "financial_stress_index": 0.5,
        "vix": 20,
        "epu_index": 125,
        "consumer_sentiment": 65,
        "treasury_spread_10y_2y": -0.4,
        "oil_price_growth": -6,
        "recession_dummy": 0,
        "election_year": 0,
        "president_party": "Democratic",
        "political_pressure_index": 0,
        "chair_leaning": "centrist",
    },
    "sc3": {
        "date": "Scenario C",
        "scenario_theme": "mixed inflation-employment tradeoff",
        "fed_funds": 5.00,
        "gdp_growth": 1.8,
        "pce_inflation": 3.1,
        "cpi_inflation": 3.4,
        "unemployment_rate": 4.9,
        "output_gap": -0.5,
        "inflation_expectation_5y": 2.4,
        "michigan_1y_expected_inflation": 3.0,
        "financial_stress_index": 0.4,
        "vix": 18,
        "epu_index": 115,
        "consumer_sentiment": 70,
        "treasury_spread_10y_2y": -0.1,
        "oil_price_growth": 4,
        "recession_dummy": 0,
        "election_year": 0,
        "president_party": "Democratic",
        "political_pressure_index": 0,
        "chair_leaning": "centrist",
    },
}


# ---------- 8. Model API call with JSON parse error handling ----------
def call_model(model_config: dict, sys_text: str, usr_text: str) -> tuple:
    """
    Call a configured OpenAI-compatible model API with structured JSON output.
    Returns (parsed_dict, usage_object); returns (None, None) on failure.
    """
    client = OpenAI(
        api_key=model_config["api_key"],
        base_url=model_config["base_url"],
    )
    try:
        res = client.chat.completions.create(
            model=model_config["model"],
            messages=[
                {"role": "system", "content": sys_text},
                {"role": "user", "content": usr_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        raw_json = res.choices[0].message.content
        usage = res.usage

        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as e:
            print(f"[JSON Parse Error] Raw output:\n{raw_json}\nError: {e}")
            return None, None

        return parsed, usage

    except Exception as e:
        print(f"[API Call Failed] Type: {type(e).__name__}  Details: {e}")
        return None, None


RAW_OUTPUT_FILE = "q1_multivariable_deepseek_qwen_raw.csv"
SUMMARY_OUTPUT_FILE = "q1_multivariable_deepseek_qwen_summary.csv"
FREQ_OUTPUT_FILE = "q1_multivariable_deepseek_qwen_frequencies.csv"
RESUME_KEY_COLUMNS = ["provider", "group", "persona", "scenario_id", "run_id"]


def make_resume_key(row: dict) -> tuple:
    return tuple(row[col] for col in RESUME_KEY_COLUMNS)


def load_existing_results(raw_output_file: str) -> tuple:
    """
    Load completed calls from a previous interrupted run.
    Returns (output_rows, completed_keys).
    """
    if not os.path.exists(raw_output_file):
        return [], set()

    df_existing = pd.read_csv(raw_output_file)
    output_rows = df_existing.to_dict("records")
    completed_keys = set()

    for row in output_rows:
        if all(col in row for col in RESUME_KEY_COLUMNS):
            row["run_id"] = int(row["run_id"])
            completed_keys.add(make_resume_key(row))

    print(
        f"Resume mode: loaded {len(output_rows)} existing records from "
        f"{raw_output_file}"
    )
    return output_rows, completed_keys


# ---------- 9. Experiment group configurations ----------
# Each group defines its name, the list of personas to iterate over
# (or None placeholders for groups without persona dimension),
# and how to label the persona in the output.
EXPERIMENT_GROUPS = [
    {
        "group": "policy_persona",
        "personas": ["hawk", "dove", "centrist"],
    },
    {
        "group": "no_persona",
        "personas": [None],  # single loop, no persona
    },
    {
        "group": "placebo",
        "personas": ["placebo"],  # single persona label for output
    },
]


# ---------- 10. Main data collection loop ----------
def collect_q1_experiment():
    """
    Iterate over models, experiment groups, personas, scenarios, and repeats.
    Export raw results and summary tables for robustness checks.
    """
    output_rows, completed_keys = load_existing_results(RAW_OUTPUT_FILE)

    total_calls = 0
    for model_config in MODEL_CONFIGS:
        for eg in EXPERIMENT_GROUPS:
            total_calls += len(eg["personas"]) * len(SCENARIOS) * N_REPEATS

    call_count = len(completed_keys)

    for model_config in MODEL_CONFIGS:
        provider = model_config["provider"]
        model_name = model_config["model"]

        for eg in EXPERIMENT_GROUPS:
            group_name = eg["group"]
            personas = eg["personas"]

            for persona in personas:
                for sc_name, sc_data in SCENARIOS.items():
                    for run_id in range(1, N_REPEATS + 1):
                        sys_prompt = build_system_prompt(group_name, persona)
                        user_prompt = USER_PROMPT_TPL.format(**sc_data)
                        persona_label = persona if persona is not None else "none"
                        resume_key = (
                            provider,
                            group_name,
                            persona_label,
                            sc_name,
                            run_id,
                        )

                        if resume_key in completed_keys:
                            continue

                        call_count += 1

                        print(
                            f"[{call_count}/{total_calls}] "
                            f"provider={provider}  model={model_name}  "
                            f"group={group_name}  persona={persona_label}  "
                            f"scenario={sc_name}  run={run_id}/{N_REPEATS}"
                        )

                        parsed, usage = call_model(model_config, sys_prompt, user_prompt)

                        if parsed is None:
                            print("  Skipping (parse or API failure)")
                            continue

                        bps = parsed.get("rate_decision_bps")
                        if bps not in [-50, -25, 0, 25, 50]:
                            print(
                                f"  Warning: rate_decision_bps={bps} "
                                f"outside allowed range [-50,-25,0,25,50]"
                            )

                        row = {
                            "provider": provider,
                            "model": model_name,
                            "group": group_name,
                            "persona": persona_label,
                            "run_id": run_id,
                            "scenario_id": sc_name,
                            "scenario_label": sc_data["date"],
                            "scenario_theme": sc_data["scenario_theme"],
                            "rate_decision_bps": bps,
                            "target_rate_level": parsed.get("target_rate_level"),
                            "primary_concern": parsed.get("primary_concern"),
                            "rationale": parsed.get("rationale"),
                            "confidence": parsed.get("confidence"),
                            "total_tokens": usage.total_tokens if usage else None,
                            **sc_data,
                        }
                        output_rows.append(row)
                        completed_keys.add(resume_key)
                        pd.DataFrame(output_rows).to_csv(
                            RAW_OUTPUT_FILE, index=False, encoding="utf-8"
                        )

                        print(
                            f"  Done  bps={bps:+d}  "
                            f"tokens={usage.total_tokens if usage else 'N/A'}"
                        )

    # ---------- 11. Export raw CSV ----------
    df_result = pd.DataFrame(output_rows)

    df_result.to_csv(RAW_OUTPUT_FILE, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("RAW DATA COLLECTION COMPLETE")
    print(f"Total records: {len(df_result)}")
    print(f"Output file: {RAW_OUTPUT_FILE}")
    print("=" * 60 + "\n")

    if df_result.empty:
        print("No successful records yet. Check API keys/network and rerun.")
        return df_result

    # ---------- 12. Summary statistics ----------
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    df_result["rate_decision_bps"] = pd.to_numeric(
        df_result["rate_decision_bps"], errors="coerce"
    )

    print("\n--- Mean & Standard Deviation of rate_decision_bps ---")
    print(
        f"{'provider':<10} {'group':<18} {'persona':<10} "
        f"{'scenario':<32} {'N':>4} {'mean':>8} {'std':>8}"
    )
    print("-" * 100)

    summary_groups = df_result.groupby(
        ["provider", "model", "group", "persona", "scenario_id"], dropna=False
    )
    summary_stats = []

    for (provider, model, grp, pers, sc), sub in summary_groups:
        bps_vals = sub["rate_decision_bps"].dropna()
        n = len(bps_vals)
        mean_val = bps_vals.mean()
        std_val = bps_vals.std(ddof=1) if n > 1 else 0.0
        summary_stats.append(
            {
                "provider": provider,
                "model": model,
                "group": grp,
                "persona": pers,
                "scenario_id": sc,
                "scenario_theme": sub["scenario_theme"].iloc[0],
                "N": n,
                "mean_bps": round(mean_val, 2),
                "std_bps": round(std_val, 2),
            }
        )
        print(
            f"{provider:<10} {grp:<18} {str(pers):<10} "
            f"{sc:<32} {n:>4} {mean_val:>+8.2f} {std_val:>8.2f}"
        )

    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv(SUMMARY_OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nSummary saved to {SUMMARY_OUTPUT_FILE}")

    # ---- 12b. Frequency distribution by model/group/persona/scenario ----
    print("\n--- Frequency Distribution of rate_decision_bps ---")
    ALLOWED_BPS = [-50, -25, 0, 25, 50]

    freq_rows = []
    for (provider, model, grp, pers, sc), sub in summary_groups:
        bps_counts = sub["rate_decision_bps"].value_counts()
        row = {
            "provider": provider,
            "model": model,
            "group": grp,
            "persona": pers,
            "scenario_id": sc,
            "scenario_theme": sub["scenario_theme"].iloc[0],
        }
        for b in ALLOWED_BPS:
            row[f"bps_{b:+d}"] = int(bps_counts.get(b, 0))
        freq_rows.append(row)

    df_freq = pd.DataFrame(freq_rows)
    df_freq = df_freq.sort_values(["provider", "group", "persona", "scenario_id"])

    col_headers = (
        f"{'provider':<10} {'group':<18} {'persona':<10} {'scenario':<32} "
        + " ".join([f"{'bps_'+str(b):>7}" for b in ALLOWED_BPS])
    )
    print(col_headers)
    print("-" * len(col_headers))
    for _, r in df_freq.iterrows():
        freq_str = " ".join([f"{int(r[f'bps_{b:+d}']):>7}" for b in ALLOWED_BPS])
        print(
            f"{r['provider']:<10} {r['group']:<18} {str(r['persona']):<10} "
            f"{r['scenario_id']:<32} {freq_str}"
        )

    df_freq.to_csv(FREQ_OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nFrequency table saved to {FREQ_OUTPUT_FILE}")

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)

    # ---------- 13. Preview raw data ----------
    try:
        display(df_result.head(20))
    except NameError:
        print(df_result.head(20))
    return df_result


# ---------- 14. Run ----------
df = collect_q1_experiment()
