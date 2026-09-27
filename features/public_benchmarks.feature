@domain
Feature: Public benchmark adapters and paired evaluation
  MinTok evaluates against recognized public benchmarks (SWE-rebench,
  SWE-Bench Pro V2, SWE-bench Multilingual) using paired same-model execution,
  strict reference-isolation, and immutable window freezing.

  Scenario: SWE-rebench raw task normalizes to standard public task
    Given a raw SWE-rebench task dictionary with instance id "0b01001001__spectree-64"
    When the task is normalized into a public benchmark task
    Then the task has repository "0b01001001/spectree"
    And the benchmark identifier is "swe-rebench"
    And the problem statement matches the issue description
    And the reference solution patch is captured in the gold metadata

  Scenario: SWE-Bench Pro V2 task normalizes with test specifications
    Given a raw SWE-Bench Pro V2 task dictionary with instance id "pallets__flask-4065"
    When the task is normalized into a public benchmark task
    Then the task has repository "pallets/flask"
    And the benchmark identifier is "swe-bench-pro-v2"
    And the fail-to-pass list contains "tests/test_basic.py::test_request"

  Scenario: Freezing a public benchmark window generates an immutable fingerprint
    Given a public benchmark window containing 3 tasks
    When the window is frozen with window name "swe-rebench-window-1"
    Then the window manifest contains an immutable SHA-256 fingerprint
    And tampering with any task prompt or commit fails fingerprint verification

  Scenario: Workspace preparation strictly isolates reference solutions
    Given a public task with repository "sample/repo" and gold patch "secret fix"
    When the task workspace is prepared at an isolated destination
    Then no gold patch or solution file exists within the workspace
    And agent tools reject paths escaping the workspace root

  Scenario: Paired public benchmark evaluation computes efficiency multiplier and distribution
    Given 10 paired public benchmark runs where control uses 80000 provider tokens and MinTok uses 20000
    When the public benchmark report is generated
    Then the control solve rate is 80%
    And the MinTok solve rate is 80%
    And the primary efficiency multiplier is 4.00x
    And the report status is "EXCELLENT"
    And the paired geometric mean savings is reported
