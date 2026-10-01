@domain
Feature: Learned MinTok Controller and PolicyBench Tournament
  To eliminate hand-written heuristics in favor of an empirical learned controller,
  MinTok trains local tabular models for solve probability, token cost, failure,
  and recovery, selecting actions via marginal token ROI and evaluating policy regret on PolicyBench.

  @domain
  Scenario: Learned action selector optimizes marginal token ROI
    Given a policy state with candidate actions "read_slice", "read_full", and "run_verifier"
    When the learned action selector evaluates marginal ROI
    Then "read_slice" achieves the highest marginal ROI ratio
    And the action maximizing marginal ROI is selected

  @domain
  Scenario: Bypass classifier detects trivial tasks to eliminate controller overhead
    Given a trivial rename task with 15 LOC and low complexity
    When the bypass classifier evaluates the task
    Then the verdict is to bypass MinTok because expected savings are less than controller overhead
    When a complex multi-file task with 250 LOC is evaluated
    Then the bypass classifier determines MinTok should be engaged

  @domain
  Scenario: Task topology classifier predicts optimal first action
    Given a task description with failing test traceback
    When the first action classifier evaluates the task
    Then the recommended first action is "TEST"
    When a task description specifies an API signature regression
    Then the recommended first action is "SEARCH"

  @domain
  Scenario: Aggressive patch stop model halts thrashing upon verification pass
    Given an applied patch with passing unit tests
    When the patch stop model evaluates the state
    Then the directive is "VERIFY_AND_STOP" with P(correct) exceeding 0.95

  @domain
  Scenario: Destructive action model flags high-risk regressions
    Given a proposed patch modifying 5 files, 120 lines, and public API
    When the destructive action model evaluates regression risk
    Then the regression risk exceeds 0.50 and review is enforced

  @domain
  Scenario: Continuous token budget model scales with uncertainty
    Given a simple localized task state
    Then continuous budget allocates 300 tokens
    Given a high uncertainty cross-module task state
    Then continuous budget scales up to 5000 tokens

  @domain
  Scenario: Model-specific policy calibration adapts to model weaknesses
    Given Model A with weak verification strength
    When the model capability adapter adjusts action weights
    Then the weight for "run_verifier" is increased by at least 1.3x

  @domain
  Scenario: Answer localization model predicts top-k files and symbols before frontier execution
    Given a task description and failing test output
    When answer localization predicts target locations
    Then top-1 file matches the traceback file and top-5 accuracy estimate exceeds 90 percent

  @domain
  Scenario: PolicyBench controller tournament evaluates regret and Pareto dominance
    Given the PolicyBench tournament evaluating 50 decision states across 6 policy arms
    When the benchmark tournament executes
    Then "Learned MinTok + OPE" achieves lower regret than "Hand-coded MinTok"
    And "Learned MinTok + OPE" is confirmed on the empirical Pareto frontier
