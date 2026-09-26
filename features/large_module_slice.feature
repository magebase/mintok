@domain
Feature: Large-module slice backend
  Control proves targeted raw source beats semantic reads on huge modules
  (1,093 vs 2,861 tokens per solved task). The slicer must out-grep grep by
  ranking exact source regions deterministically from the IR and handing
  the agent file:line excerpts — never generated prose. Hard budgets keep
  "helpfulness" from turning into speculative overfetch, the packet
  failure mode.

  Scenario: The named function's definition ranks first with its exact span
    Given a repository with files:
      """
      # file: src/big.py
      def parse_options(opts):
          if not opts:
              raise ValueError("empty")
          return opts

      def handler():
          return parse_options("x")

      def unrelated():
          return 42
      """
    And an instruction "parse_options must reject empty opts"
    When the task is sliced
    Then the first region is "src/big.py" lines 1-4 defining "parse_options"
    And the package does not define "unrelated"

  Scenario: Direct callers appear as ranked regions
    Given a repository with files:
      """
      # file: src/big.py
      def parse_options(opts):
          return opts

      def handler():
          return parse_options("x")

      def wrapper():
          return parse_options("y")
      """
    And an instruction "parse_options must reject empty opts"
    When the task is sliced
    Then the package lists "handler" under "relevant callers"
    And the package lists "wrapper" under "relevant callers"

  Scenario: Test references get their own section
    Given a repository with files:
      """
      # file: src/big.py
      def parse_options(opts):
          return opts

      # file: tests/test_big.py
      def test_parse():
          assert parse_options("a") == "a"
      """
    And an instruction "parse_options must reject empty opts"
    When the task is sliced
    Then the package lists "test_parse" under "relevant tests"

  Scenario: Self-attribute writes name the field and the writing method
    Given a repository with files:
      """
      # file: src/conf.py
      class Config:
          def __init__(self):
              self.timeout = 5

          def bump(self):
              self.timeout = 30
      """
    And an instruction "Config.bump should validate before assigning timeout"
    When the task is sliced
    Then the first region defines "bump"
    And the package lists a "Config.timeout" write

  Scenario: Supplementary regions respect the initial token budget
    Given a repository with files:
      """
      # file: src/big.py
      def parse_options(opts):
          return opts

      def caller_a():
          return parse_options("aaaaaaaa")

      def caller_b():
          return parse_options("bbbbbbbb")

      def caller_c():
          return parse_options("cccccccc")
      """
    And an instruction "parse_options must reject empty opts"
    When the task is sliced with an initial budget of 40 tokens
    Then the package fits within 40 tokens
    And the package flags truncation
    And the definition of "parse_options" is still present

  Scenario: A definition larger than the initial budget expands the budget
    Given a repository with files:
      """
      # file: src/big.py
      def parse_options(opts):
          # one
          # two
          # three
          # four
          # five
          # six
          # seven
          # eight
          # nine
          # ten
          # eleven
          # twelve
          return opts
      """
    And an instruction "parse_options must reject empty opts"
    When the task is sliced with an initial budget of 40 tokens
    Then the budget was expanded to 200 tokens
    And the definition of "parse_options" is still present

  Scenario: The package is regions and excerpts, never prose
    Given a repository with files:
      """
      # file: src/big.py
      def parse_options(opts):
          return opts
      """
    And an instruction "parse_options must reject empty opts"
    When the task is sliced
    Then every non-excerpt line is a section label or a region
