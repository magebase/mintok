@domain
Feature: Experiment funnel
  A 120-task frontier run is the final exam, not the development loop. The
  funnel screens ideas before frontier compute is spent: static integrity
  checks first, a FAST-8 smoke suite next, then widening confirmation stages
  with sequential stopping, cached controls, and adaptive concurrency.

  Scenario: The FAST-8 suite covers the discriminating behavior classes
    Given the full task set contains the FAST-8 tasks
    When the FAST-8 suite is selected from the full task set
    Then it contains exactly 8 tasks
    And it covers a simple lookup task
    And it covers a cross-file bug task
    And it covers an api signature propagation task
    And it covers a schema or framework change task
    And it covers a large file navigation task
    And it covers a simple api extension task
    And it covers the semantic trap
    And it covers a feature addition task
    And every selected task exists in the full task set

  Scenario: A dominated arm is killed at the first checkpoint
    Given 5 paired tasks where the arm uses 1.5x the control tokens and matches solve rate
    When the sequential verdict is computed after the 5th pair
    Then the arm is killed
    And the reason mentions token ratio

  Scenario: A marginal arm survives early checkpoints but dies at 15
    Given 10 paired tasks where the arm uses 1.10x the control tokens and matches solve rate
    When the sequential verdict is computed after the 10th pair
    Then the arm continues
    When the arm reaches 15 paired tasks at 1.10x the control tokens
    When the sequential verdict is computed after the 15th pair
    Then the arm is killed
    And the reason mentions no meaningful improvement

  Scenario: A clearly better arm is never killed
    Given 15 paired tasks where the arm uses 0.70x the control tokens and matches solve rate
    When the sequential verdict is computed after the 15th pair
    Then the arm continues

  Scenario: A solve-rate regression kills the arm regardless of tokens
    Given 5 paired tasks where the arm uses 0.60x the control tokens and solves 20 points fewer
    When the sequential verdict is computed after the 5th pair
    Then the arm is killed
    And the reason mentions solve rate

  Scenario: Identical control configurations reuse the cached trajectory
    Given a completed control run for task "t1" with model "m" and effort "high"
    When a control run for the same task, model, effort, harness, and toolset is requested
    Then the cache serves the stored record without a new run
    When a control run for the same task with model "other" is requested
    Then the cache reports a miss

  Scenario: Immutable task fingerprints reject drift before any run
    Given a fingerprinted task with instruction, checker, and fixture files
    When the same task is verified against its fingerprint
    Then verification passes
    When the checker text is edited and the task is verified again
    Then verification fails with a drift error naming the changed part

  Scenario: The adaptive pool stays inside the provider budget
    When a fresh adaptive pool is created
    Then it allows 8 concurrent workers
    When three clean waves complete without throttling
    Then it allows 12 concurrent workers
    When the provider throttles a wave
    Then it allows 9 concurrent workers
    And it never exceeds 12 or drops below 4

  Scenario Outline: The funnel widens only on survivors
    Given an arm with verdict history "<history>"
    When the funnel advances the arm
    Then the arm is in phase "<phase>"

    Examples:
      | history            | phase       |
      | fresh              | smoke8      |
      | smoke8_pass        | dev15       |
      | dev15_pass         | confirm30   |
      | confirm30_pass     | eval120     |
      | smoke8_killed      | dead        |
      | dev15_killed       | dead        |
      | confirm30_killed   | dead        |
