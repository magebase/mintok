@domain
Feature: Multi-stage promotion funnel and offline development gates
  MinTok 3.1 eliminates 90%+ of candidate policies locally before spending frontier compute
  via a multi-stage promotion funnel: trajectory replay, PolicyBench state evaluation,
  catastrophe regression gating, mechanism benchmarks, and FAST-12 tournament evaluation.

  Scenario: Executing Stage 1 historical trajectory replay
    Given a recorded trajectory with 4 tool turns
    When the trajectory is replayed through the offline replayer
    Then the replayed compression ratio exceeds 2.0x
    And the context rent is quantified in token-turns
    And the next-action invariance rate is at least 80%

  Scenario: Extracting and evaluating PolicyBench intermediate states
    Given a multi-turn trajectory with eventual solve
    When intermediate decision states are extracted into PolicyBench
    And a candidate local controller evaluates the states
    Then the action agreement rate is reported
    And the mean utility regret is under 0.20
    And ground truth targets for expansion and 50k horizon are evaluated

  Scenario: Executing the 10-case catastrophe regression suite
    When the catastrophe regression suite is executed
    Then all 10 catastrophe checks pass
    And zero regressions are detected

  Scenario: Running isolated mechanism benchmarks and next-action invariance
    When the isolated mechanism benchmarks are run
    Then tool virtualization gross compression exceeds 2.0x
    And state compiler compaction ratio exceeds 3.0x
    And next-action invariance exceeds 85%

  Scenario: Evaluating FAST-12 promotion tournament with sequential stopping
    Given candidate runs that save 25% tokens with identical solves on FAST-12
    When the tournament promotion evaluation is performed
    Then the tournament verdict is "PROMOTE"
    And the mean utility delta is positive

  Scenario: Running dev-eval multi-stage promotion gate via CLI
    When running "mintok dev-eval" via the CLI
    Then the CLI stdout contains "MinTok Dev-Eval — Multi-Stage Local Promotion Gate"
    And the CLI stdout contains "Verdict: PROCEED_TO_STAGE_4 (PASS)"
    And the CLI stdout contains "Catastrophe Regression Gate:      PASS"

  Scenario: Evaluating frozen holdout release gate with manifest checking
    Given 20 paired tasks with 2.0x yield and identical solves
    And a valid frozen evaluation manifest
    When the release evaluation gate is executed
    Then the release verdict is "RELEASE_APPROVED"
    And the 95% bootstrap confidence interval lower bound exceeds 1.0
    And the manifest validation passes

  Scenario: Managing isolated worktrees and concurrent paired execution
    Given a repository path and task identifiers
    When the worktree manager sets up isolated worktrees
    And a persistent server configuration is initialized for MINTOK_DEV_MODEL
    And the concurrent paired runner generates a balanced schedule
    Then the schedule alternates arm ordering between control and candidate
    And the worktree is cleanly reset

  Scenario: Evaluating Stage 3 local behavioral divergence gate and adaptive scheduling
    Given paired candidate and champion runs across FAST-12 tasks
    When the behavioral divergence is computed
    Then target file divergence, tool family divergence, and failure signature divergence are reported
    And the overall behavioral divergence is within the acceptable threshold
    And the adaptive task scheduler selects the next task with highest expected information gain

  Scenario: Maintaining multi-champion Pareto frontier and lineage tracking
    Given candidate policies with varying solve rates and token costs
    When the candidates update the Pareto frontier
    Then champion-economy, champion-success, and champion-balanced roles are populated
    And candidate promotion lineage is recorded

  Scenario: Rejecting synthetic fixture or counterfactual data from Tier 1 live release
    Given candidate runs containing synthetic or counterfactual markers
    And an evaluation manifest claiming release evaluation
    When the release evaluation gate is executed
    Then the release verdict is "RELEASE_BLOCKED"
    And the output renders the synthetic fixture warning banner
    And observed live evidence is strictly marked as false

  Scenario: Running automated promote pipeline via CLI
    When running "mintok promote v3" via the CLI
    Then the CLI stdout contains "MinTok Promotion Pipeline — Verdict: PROMOTED"
    And the CLI stdout contains "Pareto Classification:"

  Scenario: Distinguishing fast fixture evaluation from live behavioral execution
    When running "mintok dev-eval" via the CLI
    Then the CLI stdout contains "EXECUTION TYPE: [FIXTURE]"
    And the CLI stdout contains "Local Behavioral:        NO"
    And the CLI stdout contains "Fixture / Synthetic:     YES (SYNTHETIC FIXTURE)"
    When running "mintok dev-eval --mode behavioral --force-behavioral-success" via the CLI
    Then the CLI stdout contains "EXECUTION TYPE: [LOCAL_LIVE]"
    And the CLI stdout contains "Local Behavioral:        YES"
    And the CLI stdout contains "Real Model Generations:  48"
    And the CLI stdout contains "Fixture / Synthetic:     NO (LIVE EMPIRICAL)"

  Scenario: Requiring verified local behavioral gate before promoting to frontier tokens
    When running "mintok promote v3 --require-behavioral" via the CLI
    Then the CLI stdout contains "Verdict: REJECT_AT_DEV_GATE"
    And the CLI stdout contains "Promotion requires verified local behavioral evaluation"
    When running "mintok promote v3 --require-behavioral --force-behavioral-success" via the CLI
    Then the CLI stdout contains "Verdict: PROMOTED"
    And the CLI stdout contains "EXECUTION TYPE: [LOCAL_LIVE]"

  Scenario: Change-aware diagnostic task selection and staged escalation
    Given changed modules "virtualization, compaction"
    When change-aware diagnostic tasks are selected
    Then state compaction and log verbosity stress tasks are prioritized
    When staged escalation FAST-4 to FAST-8 to FAST-12 is executed
    Then the candidate advances through staged checkpoints

  Scenario: Tracking funnel precision recall and exploration slot routing
    Given a batch of local and frontier evaluations
    When funnel calibration metrics are computed
    Then precision, recall, and rank correlation are quantified
    And exploration slot routes candidate for audit


