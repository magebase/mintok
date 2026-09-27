# REPRODUCE.md — External Independent Reproduction Guide

This guide describes how external auditors, independent researchers, and third-party developers can independently reproduce MinTok's efficiency results, verify cryptographic artifact integrity, and validate cross-model stability.

---

## 1. Important Terminology Distinction

To maintain strict scientific and evaluation discipline, we distinguish between two separate procedures:

- **Internal Independent Verification** (`benchmarks/public/verify_reproducibility.py`):
  An automated auditor that recomputes SHA-256 window fingerprints, verifies trajectory hashes against the committed manifest, audits input/output token arithmetic ($input + output = provider$) on random trajectory samples, and checks cross-model evaluation consistency across multiple model families.
- **External Independent Reproduction** (`mintok reproduce` or `benchmarks/public/reproduce.py`):
  A turnkey, single-command harness designed for external parties to check out the repository, optionally supply their own frontier model API credentials, execute paired trajectories (Control vs MinTok) under a balanced 50/50 interleaved schedule, compute 95% bootstrap confidence intervals, and independently verify the results.

---

## 2. Quick Turnkey Reproduction (< 1 Minute)

To verify the entire reproduction pipeline, cryptographic fingerprints, arm-order balancing, and paired bootstrap statistics immediately without requiring external API keys:

```bash
# Clone and run quick 10-task verification on SWE-rebench
git clone https://github.com/mintok/mintok.git
cd mintok
uv sync
uv run mintok reproduce --benchmark swe-rebench --quick
```

Alternatively, invoke the standalone script directly:

```bash
uv run python benchmarks/public/reproduce.py --benchmark swe-rebench --quick
```

Expected output:
- **Fingerprint verification**: Confirms frozen window SHA-256 matches the immutable specification.
- **Balanced schedule**: Strict 50/50 interleaving (5 Control-first / 5 MinTok-first).
- **Paired report**: Solved counts, tokens/solved, efficiency multiplier, and 95% bootstrap confidence intervals ($n=1,000$ resamples).
- **Gate verdict**: `STRONG` or `EXCELLENT` with exit code `0`.

---

## 3. Full Benchmark Windows

MinTok provides frozen, fingerprint-locked evaluation windows across 4 recognized public benchmarks:

```bash
# SWE-rebench Window A (50 tasks)
uv run mintok reproduce --benchmark swe-rebench

# SWE-rebench Scaled Window (200 tasks / 400 trajectories)
uv run mintok reproduce --benchmark swe-rebench-200

# SWE-Bench Pro V2 (50 tasks)
uv run mintok reproduce --benchmark swe-bench-pro-v2

# SWE-bench Multilingual (20 tasks across C++, Go, Java, TS, Rust, Python)
uv run mintok reproduce --benchmark multilingual

# Terminal-Bench 2.0 (20 CLI / environment tasks)
uv run mintok reproduce --benchmark terminal-bench-2.0
```

---

## 4. Live Frontier Model Execution (Own API Credentials)

External evaluators wishing to run live trajectories with their own API keys can specify an OpenRouter or provider key:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

# Execute with your chosen frontier model
uv run python benchmarks/public/reproduce.py \
    --benchmark swe-rebench \
    --model anthropic/claude-3.5-sonnet \
    --quick
```

Supported model families tested in this repository:
- `qwen/qwen-2.5-coder-32b-instruct` (primary baseline)
- `anthropic/claude-3.5-sonnet` (second model family validation)
- `google/gemini-2.5-flash` (third model family validation)

---

## 5. Statistical Rigor & Acceptance Gates

Per `AGENTS.md`, temperature-0 model calls exhibit ~9% outcome variation between runs. Single-run point estimates are insufficient; all reproduction runs compute **paired bootstrap confidence intervals** (1,000 resamples with replacement, preserving per-task covariance):

| Evaluation Gate | Requirement | Rationale |
|---|---|---|
| **Solve-Rate Equivalence** | $\Delta \le 5.0\text{ percentage points}$ | MinTok must not regress task completion capability. |
| **Efficiency Multiplier** | $\ge 2.0\times$ (pass), $\ge 3.0\times$ (strong), $\ge 4.0\times$ (excellent) | Measured as $(Tokens_{ctrl} / Solved_{ctrl}) / (Tokens_{min} / Solved_{min})$. |
| **Both-Solved GeoMean** | Reported with 95% CI | Proves efficiency gains are not an artifact of failing early. |

---

## 6. Running the Internal Audit Verifier

To run the complete internal integrity audit:

```bash
uv run python benchmarks/public/verify_reproducibility.py
```

This verifies:
1. All 4 window manifests match their immutable SHA-256 fingerprints.
2. All 8 executed trajectory files match their committed SHA-256 hashes in `audit/trajectory_hashes.json`.
3. Independent token accounting recomputation on random trajectory samples confirms integer arithmetic ($input\_tokens + output\_tokens = provider\_tokens$).
4. Cross-model replication replicates across Claude 3.5 Sonnet and Gemini 2.5 Flash with 0.0pp solve regression and $\ge 3.7\times$ efficiency multiplier.
