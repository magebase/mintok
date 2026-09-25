@domain
Feature: Optimization Compute Units
  Usage is metered in Optimization Compute Units (OCU): a documented,
  deterministic transformation of total blended cost. OCUs are never a token
  markup; customers keep their own model keys and pay providers directly.

  Scenario: A run meters into OCUs
    Given a meter over 1 run(s) at $0.30 frontier, $0.02 local, $0.05 indexing, $0.01 storage, and $0.02 cpu per run
    When the metering is computed
    Then the total cost per run is "$0.40"
    And the metered total is "40.0" OCUs

  Scenario: OCUs accumulate across runs
    Given a meter over 3 run(s) at $0.30 frontier, $0.02 local, $0.05 indexing, $0.01 storage, and $0.02 cpu per run
    When the metering is computed
    Then the metered total is "120.0" OCUs

  Scenario: OCUs are a pure cost transformation, not a markup
    Given a meter over 1 run(s) at $0.30 frontier, $0.02 local, $0.05 indexing, $0.01 storage, and $0.02 cpu per run
    When the metering is computed
    Then frontier spend of "$1.00" meters to exactly "100.0" OCUs
