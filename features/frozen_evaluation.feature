Feature: MinTok-3.2-FROZEN Controller, Model Transfer, and Budget Frontier Curve

  As an AI research engineer
  I want an immutable frozen controller snapshot, unseen model transfer benchmark, and solve-vs-budget frontier curve
  So that MinTok's performance is demonstrably frozen, zero-shot generalizable, and robust under severe token starvation.

  @domain
  Scenario: Immutable MinTok-3.2-FROZEN snapshot manifest and integrity verification
    Given an immutable MinTok-3.2-FROZEN controller snapshot
    When inspecting the frozen snapshot manifest
    Then the default operating mode is Lean Core
    And the stopping threshold lambda is exactly 0.00002
    And the controller integrity hash is verified with SHA-256
    And Lean Core operates with decision latency under 0.01 milliseconds

  @domain
  Scenario: Lean Core default policy minimizes surface while retaining core gains
    Given a decision state requiring action selection
    When Lean Core evaluates the state with a confident hypothesis
    Then Lean Core applies a surgical AST patch without invoking the 7-specialist router
    And when the state has an applied patch
    Then Lean Core immediately selects verification
    And when Full Mixture is explicitly requested
    Then the controller routes through the appropriate specialist policy

  @domain
  Scenario: Zero-shot model transfer to unseen model architecture
    Given a cross-model transfer benchmark with trained models A, B, C and unseen model D
    When evaluating zero-shot transfer performance on unseen model D
    Then MinTok achieves at least 75 percent token savings on model D without retraining
    And MinTok achieves an absolute solve rate gain of at least 15 percent on model D
    And the transfer stability score is at least 0.90

  @domain
  Scenario: Solve rate versus token budget frontier curve and brutal 2,000-token cap
    Given the empirical solve rate versus token budget benchmark across 7 budget tiers
    When evaluating the frontier curve from 2,000 to 20,000 tokens
    Then MinTok shifts the integral solve capacity AUC upward by over 100 percent vs Control
    And under the brutal 2,000-token hard cap MinTok Lean achieves at least 40 percent solve
    And under the brutal 2,000-token hard cap Control achieves under 15 percent solve

  @integration
  Scenario: CLI freeze command outputs immutable snapshot manifest
    When the user runs mintok freeze with json format
    Then the CLI exits with code 0
    And the manifest confirms MinTok-3.2-FROZEN version and Lean Core mode

  @integration
  Scenario: CLI budget-curve command outputs frontier curve and brutal cap test
    When the user runs mintok budget-curve with json format
    Then the CLI exits with code 0
    And the brutal cap test reports passed with 2,000 token limit

  @integration
  Scenario: CLI model-transfer command outputs cross-model generalization report
    When the user runs mintok model-transfer with json format
    Then the CLI exits with code 0
    And the report confirms zero-shot generalization on model D
