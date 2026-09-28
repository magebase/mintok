@domain
Feature: Adaptive sequential escalation controller
  MinTok 2.0 optimizes successful tasks per million provider tokens by starting
  in a tight semantic slice and progressively unlocking context and fallback
  tooling when stagnation or context blindness is detected.

  Scenario: Feature extraction from trajectory events
    Given an empty trajectory
    When a tool "read" fails with "error: no such file foo.py"
    And a tool "slice" succeeds with 120 tokens and output "target candidates:\n  foo.py:1"
    And a tool "patch" fails with "rejected: file would not parse: syntax error"
    Then the trajectory state has 1 missing file read
    And the trajectory state has 1 rejected patch
    And the turns elapsed is 3

  Scenario: Identical test failure hashes detect stagnation
    Given an empty trajectory
    When a test failure occurs with output:
      """
      FAILED tests/test_parser.py::test_attr - AssertionError: expected 1 got 2
      """
    And another test failure occurs with output:
      """
      FAILED tests/test_parser.py::test_attr - AssertionError: expected 1 got 2
      """
    Then the consecutive test failure count is 2
    And stagnation is detected

  Scenario: Different test failures indicate forward progress
    Given an empty trajectory
    When a test failure occurs with output:
      """
      FAILED tests/test_parser.py::test_attr - AssertionError: expected 1 got 2
      """
    And another test failure occurs with output:
      """
      FAILED tests/test_parser.py::test_attr - TypeError: cannot unpack non-iterable
      """
    Then the consecutive test failure count is 1
    And stagnation is not detected

  Scenario: Traceback parsing extracts failing target frame
    Given a test failure traceback:
      """
      Traceback (most recent call last):
        File "src/scim2_filter_parser/parser.py", line 105, in parse
          raise ValueError("unexpected token")
      ValueError: unexpected token
      """
    When the traceback is parsed
    Then the failing target is "src/scim2_filter_parser/parser.py:105"

  Scenario: Escalation ladder unlocks targeted discovery on missing file reads
    Given an escalation controller at level 0
    When 2 missing file reads occur
    Then the escalation controller level is at least 3
    And the active tools include "find_files"

  Scenario: Escalation ladder unlocks full fallback on repeated failure stagnation
    Given an escalation controller at level 0
    When 2 consecutive identical test failures occur
    Then the escalation controller level is 4
    And the active tools include "shell"

  Scenario: Exploratory stalling without edits detects stagnation
    Given an escalation controller at level 0
    When 8 consecutive read queries occur without edits
    Then stagnation is detected
    And the escalation controller level is 4
    And the active tools include "shell"

  Scenario: Empty slices unlock broaden and targeted discovery
    Given an escalation controller at level 0
    When 2 empty slice queries occur
    Then the escalation controller level is at least 3
    And the active tools include "find_files"

  Scenario: Yield per million tokens accurately computes economic efficiency
    Given 4 solved tasks out of 30 attempts
    And a total spend of 11000000 provider tokens
    When economic efficiency is computed
    Then the yield is 0.36 solves per million tokens

  @integration
  Scenario: Agent CLI with adaptive policy gates shell until stagnation
    Given a repository file "src/app.py":
      """
      def hello():
          return "hello"
      """
    And an empty trajectory log "run.jsonl"
    When the agent CLI executes tool "shell" with args "echo test" under policy "adaptive"
    Then the agent CLI exit code is 3
    And the agent CLI output includes "locked: shell is locked at Level 0"

  @integration
  Scenario: Agent CLI with adaptive policy unlocks shell upon repeated test failures
    Given a repository file "src/app.py":
      """
      def hello():
          return "hello"
      """
    And a trajectory log "run2.jsonl" with 2 identical test failures
    When the agent CLI executes tool "shell" with args "echo hello from shell" under policy "adaptive"
    Then the agent CLI exit code is 0
    And the agent CLI output includes "hello from shell"
