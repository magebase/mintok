@integration
Feature: Semantic continuity
  The agent's learned program state persists across turns. Reads of facts the
  agent has already been shown are answered with a one-line verdict when
  nothing relevant changed, and writes report which learned symbols they
  affected. No new agent-facing operation exists: continuity lives behind the
  existing query and change surface, so the tool surface and its token cost
  stay exactly as pinned.

  Background:
    Given a repository file "pkg/mod.py":
      """
      import math

      def area(r):
          return math.pi * r * r

      def circ(r):
          return 2 * math.pi * r
      """
    And the repository is compiled

  Scenario: A learned symbol with unchanged sources answers with a verdict
    When the agent learns symbol "pkg.mod:area"
    And the agent re-queries symbol "pkg.mod:area" with unchanged sources
    Then the answer is the verdict "pkg.mod:area unchanged since learn #1"

  Scenario: A comment-only edit does not re-show the symbol
    When the agent learns symbol "pkg.mod:area"
    And file "pkg/mod.py" is replaced with:
      """
      import math

      # circles only
      def area(r):
          return math.pi * r * r

      def circ(r):
          return 2 * math.pi * r
      """
    And the repository is compiled again
    And the agent re-queries symbol "pkg.mod:area" after the edit
    Then the answer is the verdict "pkg.mod:area unchanged since learn #1"

  Scenario: A body-only change is reported without re-showing facts
    When the agent learns symbol "pkg.mod:area"
    And file "pkg/mod.py" is replaced with:
      """
      import math

      def area(r):
          return math.pi * r * r * 1.0

      def circ(r):
          return 2 * math.pi * r
      """
    And the repository is compiled again
    And the agent re-queries symbol "pkg.mod:area" after the edit
    Then the answer is the verdict "pkg.mod:area interface unchanged since learn #1; body changed"

  Scenario: An interface change falls through to a full show
    When the agent learns symbol "pkg.mod:area"
    And file "pkg/mod.py" is replaced with:
      """
      import math

      def area(r, scale=1):
          return math.pi * r * r * scale

      def circ(r):
          return 2 * math.pi * r
      """
    And the repository is compiled again
    And the agent re-queries symbol "pkg.mod:area" after the edit
    Then the answer shows the full facts for "pkg.mod:area"
    And the learned record for "pkg.mod:area" is updated

  Scenario: A removed symbol is answered as gone
    When the agent learns symbol "pkg.mod:circ"
    And file "pkg/mod.py" is replaced with:
      """
      import math

      def area(r):
          return math.pi * r * r
      """
    And the repository is compiled again
    And the agent re-queries symbol "pkg.mod:circ" after the edit
    Then the answer is the verdict "pkg.mod:circ no longer indexed (removed since learn #1)"

  Scenario: Relation sets answer with a verdict while the repo is unchanged
    When the agent learns symbol "pkg.mod:area"
    And the agent learns the callers of "pkg.mod:area"
    And the agent re-queries the callers of "pkg.mod:area" with unchanged sources
    Then the answer is the verdict "callers:pkg.mod:area unchanged since learn #2 (0 facts)"

  Scenario: Any tracked change invalidates relation sets
    When the agent learns symbol "pkg.mod:area"
    And the agent learns the callers of "pkg.mod:area"
    And file "pkg/mod.py" is replaced with:
      """
      import math

      def area(r):
          return math.pi * r * r * 1.0

      def circ(r):
          return 2 * math.pi * r
      """
    And the repository is compiled again
    And the agent re-queries the callers of "pkg.mod:area" after the edit
    Then the answer shows the full facts for callers of "pkg.mod:area"

  Scenario: A write reports the learned delta
    When the agent learns symbol "pkg.mod:area"
    And the agent learns symbol "pkg.mod:circ"
    And the agent changes "pkg.mod:circ" to:
      """
      def circ(r):
          return 2 * math.pi * r * 1.0
      """
    Then the write delta reports "body changed: pkg.mod:circ"
    And the write delta reports "1 learned unchanged"

  Scenario: A write that changes an interface updates the learned record
    When the agent learns symbol "pkg.mod:circ"
    And the agent changes "pkg.mod:circ" to:
      """
      def circ(r, turns=1):
          return 2 * math.pi * r * turns
      """
    Then the write delta reports "interface changed: pkg.mod:circ"
    And the learned record for "pkg.mod:circ" carries the new interface

  Scenario: Learned state round-trips through JSON
    When the agent learns symbol "pkg.mod:area"
    And the learned state is serialized and restored
    And the agent re-queries symbol "pkg.mod:area" with unchanged sources
    Then the answer is the verdict "pkg.mod:area unchanged since learn #1"
