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
