@domain
Feature: Agent Efficiency Benchmark
  The benchmark holds the agent, model, and grader constant and compares two
  arms on the same paired tasks. The score is cache-aware dollars per solved
  task, and raw traces are published so efficiency claims are independently
  verifiable. The runner and statistics are open; the optimizer under test is
  pluggable.

  Scenario: Paired arms produce cost per solved task and a work multiple
    Given these paired benchmark results:
      | task | baseline_solved | baseline_usd | optimizer_solved | optimizer_usd |
      | t1   | yes             | 2.00         | yes              | 0.50          |
      | t2   | yes             | 2.00         | yes              | 0.50          |
      | t3   | yes             | 1.00         | yes              | 0.25          |
    When the benchmark report is computed
    Then baseline solves 3 of 3 at $1.67 per solved task
    And optimizer solves 3 of 3 at $0.42 per solved task
    And the work-per-dollar multiple is 4.00
    And the report does not flag a success regression

  Scenario: A success-rate drop is flagged even when cheaper
    Given these paired benchmark results:
      | task | baseline_solved | baseline_usd | optimizer_solved | optimizer_usd |
      | t1   | yes             | 1.00         | no               | 0.10          |
      | t2   | yes             | 1.00         | yes              | 0.40          |
    When the benchmark report is computed
    Then the report flags a success regression

  Scenario: Savings summary proves the value in dollars
    Given the baseline arm spent $38.41 and the optimizer arm spent $14.03, both solving 140 of 147 tasks
    When the savings summary is rendered
    Then it reports saved "$24.38"
    And it reports reduction "63.5%"
    And it reports a "2.74x" work-per-dollar multiple

  @integration
  Scenario: The benchmark runner writes raw traces for independent verification
    Given a synthetic agent runner that solves every task
    When a 2-task benchmark runs both arms into a temporary output directory
    Then the output directory holds a trace file per arm
    And each trace records task, tokens, cost, and solved as JSON lines

  @integration
  Scenario: The CLI renders a benchmark comparison from run records
    Given run-record files for a baseline and an optimizer arm
    When I run the CLI with "benchmark {repo}/baseline.jsonl {repo}/optimizer.jsonl"
    Then the CLI exits with code 0
    And the CLI output includes "work/$"
    And the CLI output includes "4.00x"
