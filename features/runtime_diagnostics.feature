@domain
Feature: Runtime diagnostics, preflight doctor, and trajectory inspector
  MinTok 3.1 provides runtime diagnostic utilities to verify environment readiness
  and inspect trajectory economics, itemized waste taxes, and call elimination.

  Scenario: Preflight doctor check on repository environment
    Given a clean repository with structure:
      """
      pyproject.toml
      src/core/__init__.py
      src/core/math.py
      tests/test_math.py
      """
    When running "mintok doctor" on the repository
    Then the doctor report passes
    And the report includes tool ABI budget verification under 300 tokens
    And the report confirms policy compatibility for "v3"

  Scenario: Preflight doctor check rejects legacy development policy
    Given a clean repository with structure:
      """
      pyproject.toml
      src/core/__init__.py
      """
    When running "mintok doctor" with policy "adaptive"
    Then the doctor report fails
    And the report flags policy "adaptive" as obsolete

  Scenario: Inspecting run trajectory via CLI
    Given a trajectory JSON file with 3 turns
    When running "mintok inspect-run" on the trajectory
    Then the CLI stdout contains "ECONOMIC TRACE"
    And the CLI stdout contains "ITEMIZED WASTE TAXES"
    And the CLI stdout contains "FRONTIER CALL ELIMINATION"
    And the CLI stdout contains "Context Residency Tax:"
    And the CLI stdout contains "Dead Token Ratio:"
