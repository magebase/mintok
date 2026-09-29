@domain
Feature: Tool-output virtualization and observation store
  MinTok 3.0 keeps the shell and all developer tools always available while
  virtualizing voluminous outputs at the transport layer into compact semantic
  summaries with content-addressable, recoverable observation handles.

  Scenario: Large generic tool output is virtualized with observation handle
    Given an observation store with threshold of 200 characters
    When a command "ls -la" produces 1000 characters of output
    Then the virtualized output length is less than 400 characters
    And the virtualized output contains an observation handle "obs:"
    And the observation store can retrieve the original output by handle

  Scenario: Identical tool output returns deduplicated handle
    Given an observation store with threshold of 200 characters
    When a command "pytest" produces 500 characters of output
    And the identical command output is processed again
    Then the second output starts with "unchanged: obs:"
    And the handle matches the first observation

  Scenario: Pytest output is compressed to structured failure delta
    Given an observation store
    When pytest output is received:
      """
      ============================= test session starts ==============================
      collected 12 items
      tests/test_auth.py ..F..... [100%]
      =================================== FAILURES ===================================
      _________________________________ test_login ___________________________________
          def test_login():
      >       assert auth.login("admin", "wrong") is False
      E       AssertionError: assert True is False
      tests/test_auth.py:42: AssertionError
      =========================== short test summary info ============================
      FAILED tests/test_auth.py::test_login - AssertionError: assert True is False
      ========================= 1 failed, 11 passed in 0.24s =========================
      """
    Then the virtualized output has exit code 1
    And the virtualized output highlights failing test "tests/test_auth.py::test_login"
    And the virtualized output contains assertion "assert True is False"
    And the virtualized output references handle "obs:"

  Scenario: Find output summarizes package topology
    Given an observation store
    When find output is received with 150 file paths across "src/auth", "src/tokens", and "tests/auth"
    Then the virtualized output reports total files count 150
    And the virtualized output summarizes directories including "src/auth"
    And the virtualized output contains an observation handle "obs:"

  Scenario: Git diff output summarizes modified files and line deltas
    Given an observation store
    When git diff output is received:
      """
      diff --git a/mintok/auth.py b/mintok/auth.py
      index 1234567..89abcdef 100644
      --- a/mintok/auth.py
      +++ b/mintok/auth.py
      @@ -10,3 +10,5 @@ def login(user, password):
      +    if password == "wrong":
      +        return False
           return True
      """
    Then the virtualized output identifies modified file "mintok/auth.py"
    And the virtualized output reports "+2" additions
    And the virtualized output contains an observation handle "obs:"

  Scenario: Recovering observation slice by handle and filter
    Given an observation store containing:
      """
      Line 1: system initialized
      Line 2: error: token expired
      Line 3: retry attempt 1
      Line 4: error: connection refused
      Line 5: exit code 1
      """
    When the observation is expanded with filter "error:"
    Then the expanded output contains 2 lines
    And the expanded output includes "token expired"
    And the expanded output includes "connection refused"

  Scenario: Tracking net observation savings and first-class virtualization metrics
    Given an observation store
    When a command "pytest" produces 1200 characters of test failure output
    And the observation is expanded with filter "FAILED"
    Then the observation has net positive savings
    And the observation store metrics report expansion rate greater than 0
    And the net observation compression ratio is greater than 1.0

  Scenario: Multi-tier representation, recovery prediction, and action preservation
    Given an observation store
    When git status output is received:
      """
      On branch main
      Changes not staged for commit:
        modified:   mintok/router.py
        modified:   mintok/virtualization.py
      Untracked files:
        new_test.py
      """
    Then the virtualized output summarizes git status with staged 0 and unstaged 2
    And the observation tier "L0" returns only handle "obs:"
    And the predicted expansion probability for failed test is greater than 0.70
    And the compressed output preserves critical tokens with action invariance
