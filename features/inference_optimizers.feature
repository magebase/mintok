@domain
Feature: MinTok 3.1 advanced inference optimization mechanisms
  As an inference runtime optimizer
  MinTok compiles verification, delta-encodes canonical state, tracks packet utility,
  deduplicates visible source code, predicts continuation utility, and calibrates local control.

  Scenario: Verification compiler synthesizes minimal discriminating test status
    Given a failing test output with 1 failure "tests/test_calc.py::test_division" and assertion "ZeroDivisionError"
    When the verification compiler compiles the test output
    Then the verification status indicates targeted test "FAIL"
    And new regressions count is 0
    And the verification digest tokens are less than 60 tokens

  Scenario: CanonicalState delta encoding produces compact turn update
    Given an initial canonical state with goal "Fix division by zero"
    And the state has verified fact "F1" with value "division check in math.py"
    When a second state is created with added fact "F2" with value "denominator guarded"
    Then computing state delta between state 1 and state 2 produces a delta block
    And the delta text includes "+ Fact F2"
    And the delta text length is less than 200 characters

  Scenario: State residency hierarchy evicts cold dead hypotheses
    Given a canonical state with a hot goal, a hot failing assertion, and a cold rejected hypothesis
    When the state is rendered with residency filtering enabled
    Then the rendered state includes the hot failing assertion
    And the cold rejected hypothesis is archived to local store

  Scenario: Packet utility tracker measures transition value and prunes broad AST neighborhoods
    Given a packet utility tracker
    When a packet "pkt_1" of type "ast_neighborhood" with 400 tokens is evaluated
    Then the packet utility tracker recommends pruning the packet
    When a packet "pkt_2" of type "callers" with 80 tokens is evaluated
    Then the packet utility tracker recommends retaining the packet

  Scenario: Source cache deduplicates overlapping line intervals and computes novelty
    Given an empty source cache
    When a source span for "calc.py" lines 1 to 50 is registered
    Then the computed novelty is 1.0
    When an overlapping span for "calc.py" lines 20 to 60 is evaluated
    Then the computed novelty is less than 0.60
    When the exact span for "calc.py" lines 1 to 50 is registered again
    Then the source cache returns an unchanged handle "unchanged"

  Scenario: Continuation predictor triggers early abort on runaway non-progressing trajectories
    Given a continuation predictor
    When a trajectory has spent 900000 tokens across 28 turns with 6 consecutive failures
    Then the recommended continuation action is "ABORT_TASK"
    And the eventual solve probability is less than 0.05

  Scenario: Clean-context restart creates zero-baggage prompt from verified invariants
    Given a canonical state with verified facts, a current patch, and target failing tests
    When a clean-context restart prompt is constructed
    Then the prompt contains "FRESH CLEAN CONTEXT RESTART"
    And the prompt contains the current patch
    And the prompt omits dead reasoning history

  Scenario: Calibrated controller evaluates Q(s,a) and weights samples by economic regret
    Given a tabular feature dataset
    When a training sample with utility regret 0.45 is added
    Then the sample weight is 0.45
    And the calibrated controller evaluates candidate actions and selects the highest utility action
