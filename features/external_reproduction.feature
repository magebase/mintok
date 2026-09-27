@domain
Feature: External Independent Benchmark Reproduction
  MinTok provides a turnkey, one-command reproduction harness for independent
  evaluators to verify efficiency gains, solve-rate preservation, and cross-model
  stability with 95% bootstrap confidence intervals.

  Scenario: Turnkey reproduction verifies window integrity and enforces balanced schedule
    Given a request to reproduce the "swe-rebench" benchmark in quick mode
    When the external reproduction harness runs
    Then the cryptographic window fingerprint is verified
    And the execution schedule achieves exact 50/50 balance
    And the paired efficiency multiplier is at least 3.0x
    And the 95% bootstrap confidence interval lower bound exceeds 2.0x
    And the reproduction status is reported as passing

  Scenario: Cross-model replication validates stability across three model families
    Given three distinct model families "qwen-2.5-coder-32b", "claude-3-5-sonnet", and "gemini-2.5-flash"
    When cross-model paired evaluations are audited
    Then each model family maintains solve-rate difference within 5 percentage points
    And each model family achieves an efficiency multiplier of at least 3.0x
