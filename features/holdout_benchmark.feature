Feature: SWE-Holdout-150 Benchmark across Unseen Repositories

  As an AI research engineer
  I want a frozen 150-task benchmark across 6 unseen repositories evaluating Lean Core vs Control
  So that MinTok's generalizability, Solve-Adjusted Token Efficiency (SATE), and verified patch rates are proven out-of-distribution.

  @domain
  Scenario: Fresh SWE-Holdout-150 dataset loading and repository stratification
    Given the frozen SWE-Holdout-150 dataset
    When loading the 150 holdout tasks
    Then exactly 150 tasks are loaded across 6 unseen repositories
    And each unseen repository contains exactly 25 tasks
    And the unseen repositories include sqlalchemy, scikit-learn, rich, marshmallow, httpx, and tortoise-orm

  @domain
  Scenario: Paired interleaved evaluation proves SATE multiplier and token reduction
    Given the fresh SWE-Holdout-150 benchmark suite
    When executing paired interleaved evaluation between Control and MinTok Lean Core
    Then MinTok Lean Core achieves at least 80 percent token reduction vs Control
    And MinTok Lean Core achieves an absolute solve rate gain of at least 20 percent
    And MinTok Lean Core achieves at least 8 times higher Solve-Adjusted Token Efficiency SATE
    And MinTok Lean Core achieves a verified patch rate of at least 95 percent
    And Control achieves a verified patch rate under 85 percent

  @domain
  Scenario: Unseen repository breakdown demonstrates consistent generalization across all domains
    Given the completed SWE-Holdout-150 benchmark result
    When inspecting the breakdown across each of the 6 unseen repositories
    Then every repository demonstrates at least 80 percent token savings
    And every repository demonstrates a positive absolute solve rate gain
    And every repository achieves a SATE improvement ratio of at least 7.0x

  @domain
  Scenario: Granular token decomposition proves virtualization is not the sole driver
    Given the completed SWE-Holdout-150 benchmark result
    When inspecting the granular token decomposition and mechanism attribution
    Then Output Virtualization accounts for under 60 percent of total savings
    And non-virtualization mechanisms account for over 40 percent of total savings
    And AST state compilation and turn reduction together save over 10,000 tokens per task

  @domain
  Scenario: Paired 2x2 contingency matrix and McNemar test confirm significant rescue
    Given the completed SWE-Holdout-150 benchmark result
    When inspecting the paired 2x2 contingency matrix
    Then the paired 2x2 contingency table records at least 40 Lean-only solves
    And the paired 2x2 contingency table records at least 1 Control-only solve
    And the McNemar test p-value is under 0.001
    And the odds ratio for Lean rescue exceeds 5.0

  @domain
  Scenario: 11-point information-equivalence audit confirms zero data leakage
    Given the completed SWE-Holdout-150 benchmark result
    When running the 11-point information-equivalence audit
    Then all 11 audit checks pass with zero reference or test leakage
    And the overall audit status is verified passed

  @domain
  Scenario: Cold-start versus warm-start evaluation confirms zero dependence on cached state
    Given the completed SWE-Holdout-150 benchmark result
    When comparing cold-start and warm-start evaluation
    Then cold-start achieves at least 90 percent solve rate and over 80 percent token savings
    And cold-start is confirmed independently viable without prior repository intelligence

  @integration
  Scenario: CLI bench command runs holdout-150 suite
    When the user runs mintok bench with holdout-150 suite and json format
    Then the CLI exits with code 0
    And the benchmark output reports 150 total tasks and 6 unseen repositories

  @integration
  Scenario: CLI holdout-150 command reports comprehensive SATE and repo breakdowns
    When the user runs mintok holdout-150 with json format
    Then the CLI exits with code 0
    And the output contains sate_efficiency and repo_breakdowns for all 6 repositories
