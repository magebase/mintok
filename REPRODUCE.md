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

## 4. Current September 2026 OpenRouter Free Model Matrix

External evaluators wishing to test current models can use the frozen set of verified-free OpenRouter models:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

# Execute with Qwen3.8 27B free (August 2026)
uv run python benchmarks/public/reproduce.py \
    --benchmark swe-rebench \
    --model qwen/qwen3.8-27b:free \
    --quick

# Or with Poolside Laguna S 2.1 free (July 2026)
uv run python benchmarks/public/reproduce.py \
    --benchmark swe-rebench \
    --model poolside/laguna-s-2.1:free \
    --quick
```

**Primary Free Model Set (Zero Token Pricing, Fixed Identities)**:
- `qwen/qwen3.8-27b:free` (released Aug 14, 2026; 262K context, 68.1 Coding Index)
- `poolside/laguna-s-2.1:free` (released Jul 21, 2026; 262K context, tool-calling agent)
- `nvidia/nemotron-3-ultra-550b-a55b:free` (released Jul 28, 2026; 1M context, 550B MoE agent)
- `cohere/north-mini-code:free` (released Jun 18, 2026; 256K context, SWE-trained agent)

**Current September 2026 Stress-Test**:
- `stealth/space-bunny-alpha` (released Sep 23, 2026; 1M context stealth preview)

*Important: Do NOT use `openrouter/free`, because dynamic model routing invalidates paired comparisons.*

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
1. All 5 window manifests match their immutable SHA-256 fingerprints.
2. All 20 executed trajectory files match their committed SHA-256 hashes in `audit/trajectory_hashes.json`.
3. Independent token accounting recomputation on random trajectory samples confirms integer arithmetic ($input\_tokens + output\_tokens = provider\_tokens$).
4. Cross-model replication replicates across September 2026 free models with 0.0pp solve regression and $\ge 3.4\times$ efficiency multiplier.
5. Historical cross-model replication across Claude 3.5 Sonnet and Gemini 2.5 Flash.
6. Zero-discordance and harness coupling audit: confirms 588 independent API requests (0 shared IDs), 100% separate workspaces/checkers, 100% completion text divergence, and 52.5% distinct patch implementations.
