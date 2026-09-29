@domain
Feature: Repository execution profiles and complexity priors
  MinTok 3.0 scans and caches repository execution profiles once, eliminating
  repetitive frontier-model exploration of package managers, test runners, and
  monorepo layout, and deriving repo difficulty priors to route strategy.

  Scenario: Detecting a single-package Python repository profile
    Given a repository with structure:
      """
      pyproject.toml
      src/calculator/__init__.py
      src/calculator/ops.py
      tests/test_ops.py
      """
    When the repository execution profile is scanned
    Then the detected package manager is "pyproject"
    And the detected test runner is "pytest"
    And the key directories include "src/calculator" and "tests"
    And the recommended strategy is "compressed-first"

  Scenario: Detecting a complex multi-package monorepo profile
    Given a monorepo with 5 packages:
      """
      setup.py
      azure/cli/__init__.py
      azure/cli/command_modules/vm/__init__.py
      azure/cli/command_modules/storage/__init__.py
      azure/cli/command_modules/network/__init__.py
      azure/cli/command_modules/keyvault/__init__.py
      tests/test_vm.py
      """
    When the repository execution profile is scanned
    Then the monorepo package count is at least 4
    And the complexity score exceeds 0.6
    And the recommended strategy is "virtualized-shell"

  Scenario: Rendering compact execution profile for agent context
    Given a cached repository profile for "ASFHyP3/hyp3-sdk":
      """
      language: python
      test_command: pytest -q
      packages: hyp3_sdk
      key_directories: src/hyp3_sdk, tests
      """
    When the profile context is formatted for the agent
    Then the profile token estimate is under 150 tokens
    And the profile context contains "test_command: pytest -q"
    And the profile context contains "hyp3_sdk"

  Scenario: Persistent profile serialization and reload
    Given a repository profile for "msrest"
    When the profile is serialized to JSON and reloaded
    Then the reloaded profile matches the original profile
