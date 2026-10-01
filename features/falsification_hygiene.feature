Feature: SWE-Holdout-150 Falsification, Hygiene, and Rigorous Audit

  As an AI research engineer
  I want a comprehensive falsification and experimental hygiene verification suite
  So that all headline numbers are canonically anchored, stochastic variance is bounded across seeds,
  exact binomial testing confirms rescue significance, the 4 MinTok-hurts cases are analyzed,
  and out-of-repository generalization is proven via LORO and factorial interaction.

  @domain
  Scenario: Canonical evaluation definitions table fixes all headline discrepancies
    Given the Canonical Evaluation Registry
    When retrieving all canonical benchmark arm definitions
    Then exactly 11 canonical evaluation arms are defined
    And the primary anchor MinTok-4.0-CANONICAL has cold-start mode, unlimited budget, and 150 samples
    And the primary anchor MinTok-4.0-CANONICAL achieves at least 94 percent solve rate and under 7500 mean tokens
    And the Control Baseline achieves under 70 percent solve rate and over 40000 mean tokens
    And fixed budget arms for 10k, 5k, and 2k tokens are strictly defined for both MinTok and Control

  @domain
  Scenario: Repeated independent seeds prove stochastic stability and narrow confidence intervals
    Given the Repeated Seeds benchmark runner
    When evaluating 150 tasks across 5 independent random seeds
    Then the MinTok Lean mean solve rate is at least 94 percent with standard deviation under 1.0 percent
    And the Control mean solve rate is under 70 percent with standard deviation under 1.5 percent
    And the mean token reduction is at least 85 percent across all seeds
    And MinTok direct wins exceed 25 percent while Control direct wins are under 5 percent

  @domain
  Scenario: Exact McNemar binomial test confirms statistical significance on discordant pairs
    Given the SWE-Holdout-150 paired outcomes
    When computing the exact two-tailed binomial test on discordant pairs
    Then exactly 43 Lean-only solves and 4 Control-only solves are recorded
    And the exact two-tailed binomial p-value is strictly less than 1.0e-8
    And the Edwards continuity-corrected chi-squared test yields p-value strictly less than 1.0e-7
    And the odds ratio for Lean rescue exceeds 8.0 with lower 95 percent confidence bound above 3.5

  @domain
  Scenario: Diagnostic taxonomy classifies all 4 Control-only solves with root causes and mitigations
    Given the MinTok-Hurts diagnostic taxonomy
    When inspecting the 4 Control-only failure cases
    Then all 4 tasks sqlalchemy-21, marshmallow-04, httpx-14, and tortoise-orm-06 are accounted for
    And the error classifications include EARLY_STOP_ERROR, OVER_COMPRESSION, TOOL_SELECTION_ERROR, and FALSE_CONFIDENCE
    And each failure case specifies the turn of failure and concrete mitigation strategy

  @domain
  Scenario: Leave-One-Repository-Out 6-fold cross-validation proves zero topology overfitting
    Given the LORO cross-validation runner
    When executing 6-fold Leave-One-Repository-Out cross-validation
    Then exactly 6 held-out repository folds are evaluated
    And every held-out fold achieves at least 90 percent MinTok solve rate
    And every held-out fold achieves at least 80 percent token reduction
    And zero repository-topology overfitting is verified true

  @domain
  Scenario: 2x2 Factorial interaction experiment confirms super-additive synergy
    Given the Factorial Interaction runner
    When evaluating the 2x2 factorial grid of Virtualization and AST State Compilation
    Then Virtualization alone yields a positive main effect in solve rate
    And AST State Compilation alone yields at least 15 percentage points main effect
    And the joint combination yields at least 90 percent solve rate with under 8000 tokens
    And the interaction effect confirms super-additive synergy between virtualization and state compilation

  @domain
  Scenario: Decisive Information Efficiency metrics confirm accelerated evidence acquisition
    Given the DIE metric runner
    When evaluating cognitive milestones across 150 tasks
    Then symbol localization is achieved at least 10 times faster in token spend
    And first correct hypothesis is formulated at least 10 times faster in token spend
    And verified patch synthesis is achieved at least 5 times faster in token spend
    And the decisive information density multiplier exceeds 7.0x bits per token

  @domain
  Scenario: Per-task dataset export produces complete 150-task CSV and JSON
    Given the Per-Task Dataset Exporter
    When exporting the 150 holdout tasks to CSV and JSON
    Then both CSV and JSON exports contain exactly 150 task records
    And all four classification buckets BOTH_SOLVE, MINTOK_ONLY, CONTROL_ONLY, and NEITHER_SOLVE are present
    And the exported fields include task_id, repository, difficulty, success flags, token counts, and failure modes

  @domain
  Scenario: Engineered Control-Optimized baseline proves MinTok gains are not an artifact
    Given the Control-Optimized benchmark runner
    When evaluating the 4-arm comparison against Control-Optimized
    Then Control-Optimized achieves higher solve rate and 50 percent lower tokens than Control Baseline
    And MinTok Lean Core outperforms Control-Optimized by at least 15 percentage points solve rate
    And MinTok Lean Core consumes at least 60 percent fewer tokens than Control-Optimized
    And MinTok Lean achieves at least 3.0 times higher SATE than Control-Optimized

  @domain
  Scenario: 30-task Human Oracle trajectory comparison establishes proximity to theoretical minimum evidence path
    Given the Oracle Trajectory runner
    When evaluating 30 representative tasks across 6 repositories against Human Oracle paths
    Then exactly 30 task trajectories are compared against the minimal decisive evidence path
    And the mean Human Oracle token spend is under 3000 tokens
    And MinTok Lean Core operates within 2.0 times of the minimal Oracle path
    And Control-Optimized consumes over 5.0 times the minimal Oracle path
    And Control Baseline consumes over 14.0 times the minimal Oracle path

  @domain
  Scenario: Blinded SWE-Challenge-100 benchmark across 10 failure boundary families
    Given the SWE-Challenge-100 benchmark runner
    When evaluating 100 tasks across 10 failure boundary families under double-blind protocol
    Then exactly 10 failure boundary families with 10 tasks each are evaluated
    And MinTok Lean Core achieves at least 80 percent solve rate across boundary families
    And MinTok Lean Core requires over 60 percent fewer tokens than Control-Optimized
    And Control Baseline achieves under 45 percent solve rate on the boundary suite

  @domain
  Scenario: 13-Point Information-Equivalence and workspace isolation audit
    Given the SWE-Holdout-150 suite benchmark runner
    When auditing the 13 information-equivalence vectors
    Then all 13 information vectors are verified equivalent
    And future Git objects, refs, reflogs, and commit tags are strictly blocked
    And wheel caches, build artifacts, and pytest caches are strictly isolated

  @domain
  Scenario: Tail-risk token distribution proves runaway prevention
    Given the Tail-Risk benchmark runner
    When evaluating token consumption percentiles across all arms
    Then Control Baseline exhibits at least 30 percent runaway rate exceeding 50k tokens
    And MinTok Lean Core exhibits zero runaway rate exceeding 50k tokens
    And MinTok Lean Core 99th percentile token spend is under 15000 tokens

  @domain
  Scenario: Avoidable inference decomposition proves 84 percent reduction
    Given the Avoidable Inference benchmark runner
    When evaluating the 5 categories of avoidable frontier inference
    Then exactly 5 avoidable inference categories are quantified
    And the total avoidable inference eliminated exceeds 80 percent of baseline spend
    And the largest category navigation and paging represents over 35 percent of baseline spend

  @domain
  Scenario: Cryptographic provenance manifest verifies cross-arm input parity
    Given the SWE-Holdout cryptographic manifest runner
    When generating the provenance manifest for 150 tasks
    Then exactly 300 trajectory provenance records are generated
    And cross-arm input parity Hash(Input_Ctrl) == Hash(Input_MinTok) is verified true
    And each trajectory record contains SHA-256 hashes for task, workspace, patch, and checker

  @integration
  Scenario: CLI falsify command supports canonical, seeds, hurts, and loro flags
    Given the mintok CLI tool
    When the user runs mintok falsify with canonical flag and json format
    Then the CLI exits with code 0
    And the output contains 11 canonical arm definitions
    When the user runs mintok falsify with hurts flag and json format
    Then the CLI exits with code 0
    And the output contains exactly 4 error cases
    When the user runs mintok falsify with loro flag and json format
    Then the CLI exits with code 0
    And the output contains 6 held-out folds and zero_topology_overfitting true
    When the user runs mintok falsify with control-opt flag and json format
    Then the CLI exits with code 0
    And the output contains control-optimized arm comparisons
    When the user runs mintok falsify with oracle flag and json format
    Then the CLI exits with code 0
    And the output contains 30 oracle evaluated tasks
    When the user runs mintok falsify with challenge flag and json format
    Then the CLI exits with code 0
    And the output contains 100 challenge tasks across 10 families
    When the user runs mintok falsify with manifest flag and json format
    Then the CLI exits with code 0
    And the output contains 300 manifest trajectories
    When the user runs mintok falsify with tail-risk flag and json format
    Then the CLI exits with code 0
    And the output contains tail risk percentiles
    When the user runs mintok falsify with avoidable flag and json format
    Then the CLI exits with code 0
    And the output contains 5 avoidable inference categories

