@domain
Feature: Semantic coprocessor and macro-actions
  MinTok 3.0 provides local deterministic macro-actions that orchestrate multi-step
  inspections (test execution, traceback extraction, AST symbol mapping, callers)
  in zero frontier turns, returning compact bounded evidence packets (<600 tokens).

  Scenario: Investigating a test failure produces a compact evidence packet
    Given a Python repository with a failing function in "src/service.py":
      """
      def process_data(data):
          if not data:
              raise ValueError("empty data")
          return data.upper()
      """
    And a test in "tests/test_service.py":
      """
      def test_empty():
          process_data("")
      """
    When a test failure occurs with traceback pointing to "src/service.py:3"
    And macro-action "investigate_failure" is executed
    Then the resulting evidence packet targets symbol "process_data"
    And the evidence packet target file is "src/service.py"
    And the evidence packet token count is under 400 tokens
    And the evidence packet includes the slice excerpt of "process_data"

  Scenario: Localizing a symbol bundles definition, signature, and callers
    Given a repository with symbol "authenticate_user" defined in "src/auth.py"
    And "src/api.py" calls "authenticate_user"
    When macro-action "localize_symbol" is invoked for "authenticate_user"
    Then the symbol packet identifies definition in "src/auth.py"
    And the symbol packet identifies caller "src/api.py"
    And the packet size is under 300 tokens

  Scenario: Assessing patch impact distinguishes body edit from interface change
    Given an existing symbol "format_record(record)" in "src/formatter.py"
    When a patch modifies internal logic of "format_record" without changing signature
    Then the patch assessment classifies the change as "body_only"
    And interface compatibility is preserved
