@integration
Feature: MinTok 3.0 runtime inference optimizer
  MinTok 3.0 provides always-available execution (shell, suite, patch, read,
  grep, find_files) from turn zero, virtualizes output to eliminate context
  blowups, provides macro-actions, and compiles conversational state.

  Scenario: Policy v3 allows unrestricted shell from turn 0
    Given an empty trajectory log "v3_turn0.jsonl"
    When the agent CLI executes tool "shell" with args "echo 'hello from turn 0'" under policy "v3"
    Then the agent CLI exit code is 0
    And the agent CLI output includes "hello from turn 0"

  Scenario: Policy v3 virtualizes large command output with observation handle
    Given an empty trajectory log "v3_virt.jsonl"
    When the agent CLI executes tool "shell" with args "seq 1 100" under policy "v3"
    Then the agent CLI exit code is 0
    And the agent CLI output includes "obs:"
    And the agent CLI output includes "full output → obs:"

  Scenario: Policy v3 expands stored observation handle
    Given an empty trajectory log "v3_expand.jsonl"
    When the agent CLI executes tool "shell" with args "seq 1 100 | sed 's/50/error: test failure 1/'" under policy "v3"
    Then the agent CLI output includes "obs:"
    When the latest observation handle is expanded with filter "error:" under policy "v3"
    Then the agent CLI exit code is 0
    And the agent CLI output includes "error: test failure 1"

  Scenario: Policy v3 executes investigate_failure macro-action
    Given an empty trajectory log "v3_macro.jsonl"
    And a repository with a failing function "calculate" in "calc.py"
    When the agent CLI executes macro "investigate_failure" on "calc.py:3" under policy "v3"
    Then the agent CLI exit code is 0
    And the agent CLI output includes "[evidence packet: calculate"

  Scenario: Six-arm ablation ladder gates tools according to configuration
    Given an empty trajectory log "v3_ablation.jsonl"
    And a repository with a failing function "calculate" in "calc.py"
    When the agent CLI executes tool "shell" with args "echo 'test ablation'" under policy "v3_v"
    Then the agent CLI exit code is 0
    When the agent CLI executes tool "localize_symbol" with args "calculate" under policy "v3_v"
    Then the agent CLI exit code is 3
    And the agent CLI output includes "locked: tool 'localize_symbol' not in policy v3_v"
    When the agent CLI executes tool "localize_symbol" with args "calculate" under policy "v3_vcrm"
    Then the agent CLI exit code is 0
    And the agent CLI output includes "[symbol packet: calculate"
