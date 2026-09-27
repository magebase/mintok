@domain
Feature: Large-module slice backend
  Control proves targeted raw source beats semantic reads on huge modules
  (1,093 vs 2,861 tokens per solved task). The slicer must out-grep grep by
  ranking exact source regions deterministically from the IR and handing
  the agent file:line excerpts — never generated prose. Hard budgets keep
  "helpfulness" from turning into speculative overfetch, the packet
  failure mode.

  Scenario: A class-level target is sliced by member methods, not dumped
    Given a repository with files:
      """
      # file: src/big.py
      class Wide:
          # Wide owns several padded methods

          def alpha(self, text):
              cleaned = text.strip()
              if not cleaned:
                  return "empty"
              return cleaned.lower()

          def beta(self):
              pad = "beta"
              for i in range(30):
                  pad = pad + 'x0'
                  pad = pad + 'x1'
                  pad = pad + 'x2'
                  pad = pad + 'x3'
                  pad = pad + 'x4'
                  pad = pad + 'x5'
                  pad = pad + 'x6'
                  pad = pad + 'x7'
                  pad = pad + 'x8'
                  pad = pad + 'x9'
                  pad = pad + 'x10'
                  pad = pad + 'x11'
                  pad = pad + 'x12'
                  pad = pad + 'x13'
                  pad = pad + 'x14'
                  pad = pad + 'x15'
                  pad = pad + 'x16'
                  pad = pad + 'x17'
                  pad = pad + 'x18'
                  pad = pad + 'x19'
                  pad = pad + 'x20'
                  pad = pad + 'x21'
                  pad = pad + 'x22'
                  pad = pad + 'x23'
                  pad = pad + 'x24'
                  pad = pad + 'x25'
                  pad = pad + 'x26'
                  pad = pad + 'x27'
                  pad = pad + 'x28'
                  pad = pad + 'x29'
                  pad = pad + 'x30'
                  pad = pad + 'x31'
                  pad = pad + 'x32'
                  pad = pad + 'x33'
                  pad = pad + 'x34'
                  pad = pad + 'x35'
                  pad = pad + 'x36'
                  pad = pad + 'x37'
                  pad = pad + 'x38'
                  pad = pad + 'x39'
                  pad = pad + 'x40'
                  pad = pad + 'x41'
                  pad = pad + 'x42'
                  pad = pad + 'x43'
                  pad = pad + 'x44'
              return pad

          def gamma(self):
              pad = "gamma"
              for i in range(30):
                  pad = pad + 'x0'
                  pad = pad + 'x1'
                  pad = pad + 'x2'
                  pad = pad + 'x3'
                  pad = pad + 'x4'
                  pad = pad + 'x5'
                  pad = pad + 'x6'
                  pad = pad + 'x7'
                  pad = pad + 'x8'
                  pad = pad + 'x9'
                  pad = pad + 'x10'
                  pad = pad + 'x11'
                  pad = pad + 'x12'
                  pad = pad + 'x13'
                  pad = pad + 'x14'
                  pad = pad + 'x15'
                  pad = pad + 'x16'
                  pad = pad + 'x17'
                  pad = pad + 'x18'
                  pad = pad + 'x19'
                  pad = pad + 'x20'
                  pad = pad + 'x21'
                  pad = pad + 'x22'
                  pad = pad + 'x23'
                  pad = pad + 'x24'
                  pad = pad + 'x25'
                  pad = pad + 'x26'
                  pad = pad + 'x27'
                  pad = pad + 'x28'
                  pad = pad + 'x29'
                  pad = pad + 'x30'
                  pad = pad + 'x31'
                  pad = pad + 'x32'
                  pad = pad + 'x33'
                  pad = pad + 'x34'
                  pad = pad + 'x35'
                  pad = pad + 'x36'
                  pad = pad + 'x37'
                  pad = pad + 'x38'
                  pad = pad + 'x39'
                  pad = pad + 'x40'
                  pad = pad + 'x41'
                  pad = pad + 'x42'
                  pad = pad + 'x43'
                  pad = pad + 'x44'
              return pad

          def delta(self):
              pad = "delta"
              for i in range(30):
                  pad = pad + 'x0'
                  pad = pad + 'x1'
                  pad = pad + 'x2'
                  pad = pad + 'x3'
                  pad = pad + 'x4'
                  pad = pad + 'x5'
                  pad = pad + 'x6'
                  pad = pad + 'x7'
                  pad = pad + 'x8'
                  pad = pad + 'x9'
                  pad = pad + 'x10'
                  pad = pad + 'x11'
                  pad = pad + 'x12'
                  pad = pad + 'x13'
                  pad = pad + 'x14'
                  pad = pad + 'x15'
                  pad = pad + 'x16'
                  pad = pad + 'x17'
                  pad = pad + 'x18'
                  pad = pad + 'x19'
                  pad = pad + 'x20'
                  pad = pad + 'x21'
                  pad = pad + 'x22'
                  pad = pad + 'x23'
                  pad = pad + 'x24'
                  pad = pad + 'x25'
                  pad = pad + 'x26'
                  pad = pad + 'x27'
                  pad = pad + 'x28'
                  pad = pad + 'x29'
                  pad = pad + 'x30'
                  pad = pad + 'x31'
                  pad = pad + 'x32'
                  pad = pad + 'x33'
                  pad = pad + 'x34'
                  pad = pad + 'x35'
                  pad = pad + 'x36'
                  pad = pad + 'x37'
                  pad = pad + 'x38'
                  pad = pad + 'x39'
                  pad = pad + 'x40'
                  pad = pad + 'x41'
                  pad = pad + 'x42'
                  pad = pad + 'x43'
                  pad = pad + 'x44'
              return pad

      """
    And an instruction "Wide.alpha must treat empty text as empty"
    When the task is sliced
    Then the slice includes the class header of "Wide"
    And the slice includes method "alpha"
    And the slice excludes the body of method "gamma"
    And the package fits the initial budget

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
    And the slice shows the "timeout" field write in the writing method's body

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

  Scenario: Slicer ranking has zero dependence on reference solutions or patches
    Given a repository with files:
      """
      # file: src/big.py
      def parse_options(opts):
          return opts
      def validate_options(opts):
          return bool(opts)
      """
    And an instruction "parse_options must reject empty opts"
    When the task is sliced
    Then the slice ranking is independent of external solutions or reference fixes

