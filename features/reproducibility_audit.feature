@domain
Feature: Reproducibility and Audit Verification
  MinTok enforces cryptographic immutability of public benchmark windows,
  verifies independent token accounting across executed trajectories, and
  validates that efficiency gains replicate across current September 2026 free models.

  Scenario: Audit verifies all public benchmark windows match cryptographic fingerprints
    Given the frozen public benchmark windows directory
    When an auditor verifies the window fingerprints
    Then all windows pass cryptographic verification
    And no task fingerprint has drifted

  Scenario: Independent token accounting check confirms provider token integrity
    Given 1180 executed public benchmark trajectory records
    When an independent auditor samples 15 random task trajectories
    Then the provider token total strictly equals the sum of input and output tokens
    And no trajectory contains zero or negative token values

  Scenario: Current September 2026 OpenRouter free models replicate token efficiency
    Given the September 2026 free model matrix including "qwen/qwen3.8-27b:free" and "nvidia/nemotron-3-ultra-550b-a55b:free"
    When the free model replication is audited
    Then all endpoints are verified strictly free with zero token pricing
    And every model family achieves at least 3.0x token efficiency
    And no model exhibits solve rate degradation exceeding 5 percentage points

  Scenario: Historical model family replicates token efficiency without solve regression
    Given a paired evaluation using a historical model family "claude-3-5-sonnet"
    When the cross-model benchmark is evaluated
    Then the solve rate difference is at most 5 percentage points
    And the efficiency multiplier is at least 3.0x
