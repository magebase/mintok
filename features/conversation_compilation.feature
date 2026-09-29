@domain
Feature: Conversation state compilation
  MinTok 3.0 compiles multi-turn agent conversation history into a structured,
  canonical working state at semantic checkpoints, discarding repetitive prose
  and unbounded tool chatter while preserving all critical reasoning facts.

  Scenario: Compiling initial conversation into canonical state
    Given a task goal "Fix ZeroDivisionError in calculate_ratio when denominator is zero"
    When the assistant observes file "src/math_ops.py" containing "def calculate_ratio(a, b):"
    Then the canonical state has goal "Fix ZeroDivisionError in calculate_ratio when denominator is zero"
    And the canonical state knows file "src/math_ops.py"
    And the canonical state knows symbol "calculate_ratio"

  Scenario: Test failures update current failure signatures and hypotheses
    Given an active canonical state for "Fix ZeroDivisionError"
    When the assistant adopts hypothesis "calculate_ratio fails to guard against b == 0"
    And a test suite run fails with:
      """
      FAILED tests/test_math.py::test_zero_denom - ZeroDivisionError: division by zero
      """
    Then the canonical state active hypothesis is "calculate_ratio fails to guard against b == 0"
    And the canonical state records current failure "tests/test_math.py::test_zero_denom"
    And the canonical state failure signature matches "ZeroDivisionError"

  Scenario: Verified patch turns active hypothesis into verified fact
    Given an active canonical state with hypothesis "calculate_ratio fails to guard against b == 0"
    And current failure "tests/test_math.py::test_zero_denom"
    When a patch is applied to "src/math_ops.py" adding guard "if b == 0: return 0.0"
    And a test suite run passes with 12 passed
    Then the active hypothesis is cleared
    And the verified facts include "b == 0 guard resolved tests/test_math.py::test_zero_denom"
    And the current failures list is empty

  Scenario: Rejected hypothesis is preserved to prevent looping
    Given an active canonical state with hypothesis "maybe input a is None"
    When a test shows input a is valid integer 10
    And the hypothesis is rejected
    Then the rejected hypotheses include "maybe input a is None"
    And the active hypothesis is cleared

  Scenario: Conversation history is compacted into bounded state message
    Given an active canonical state with 3 verified facts and 1 known file
    When the conversation state is rendered for the frontier model
    Then the rendered state length is under 500 characters
    And the rendered state contains section "**Verified Facts:**"
    And the rendered state contains section "**Goal:**"
