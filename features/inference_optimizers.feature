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

  Scenario: Verification safety tracker calculates miss rate
    Given a verification safety tracker
    When 20 verification events occur with 1 false pass and 5 broader failures
    Then the computed miss rate is 0.20
    And the safety gate fails when max allowed miss rate is 0.05

  Scenario: Passing test verification cache avoids redundant execution
    Given a passing verification cache
    When test "tests/test_foo.py::test_bar" passes for dependency hash "abc1234"
    Then the test is confirmed as a cached pass for hash "abc1234"
    And the test is not a cached pass for hash "def5678"

  Scenario: Context rent manager evicts leases on disproven hypothesis event
    Given a context rent manager with an admitted object associated with hypothesis "H1"
    When an event "hypothesis_rejected" for "H1" is triggered
    Then the object associated with "H1" is evicted from active memory

  Scenario: Cache-aware compaction calculates positive net benefit when replay savings exceed invalidation
    When evaluating cache compaction for prefix 50 tokens and delta 100 tokens across 15 remaining turns
    Then the net compaction benefit is positive

  Scenario: Source cache distinguishes RESIDENT and SEEN regions and supports rehydration
    Given a source cache with a resident span for "utils.py" lines 10 to 40
    When the span is evicted from active context
    Then its status becomes "SEEN"
    When the span is rehydrated
    Then its status becomes "RESIDENT"

  Scenario: Dynamic tool surface filters schema by phase
    When filtering tool surface for phase "verify"
    Then only "verify" and "query" tools are exposed
    And the tool surface tokens for "verify" are less than 80 tokens

  Scenario: Adaptive packet pruning suppresses pruning under retriever disagreement
    Given a packet utility tracker
    When a packet of type "ast_neighborhood" with 500 tokens has retriever disagreement
    Then the packet utility tracker recommends retaining the packet

  Scenario: Token usage breakdown computes exact billed provider dollars
    Given a token usage breakdown with 100000 fresh, 400000 cached, 20000 output, and 10000 reasoning tokens
    When computing billed cost under default pricing
    Then the total billed cost is greater than 0.85 and less than 0.95 dollars

  Scenario: Lost solve attribution classifies failure causes
    When attributing a lost solve where the task was early stopped
    Then the attribution category is "premature_early_stop"

