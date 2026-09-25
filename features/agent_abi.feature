Feature: Agent ABI
  Frontier agents learn one compact interface over the semantic IR instead of
  dozens of verbose tools and raw source reads.

  Background:
    Given a repository file "billing.py":
      """
      MAX_RETRIES = 3


      class Gateway:
          def refund(self, payment_id, amount):
              return True


      class Payment:
          def refund(self, amount: int) -> bool:
              '''Refund part or all of a captured payment.'''
              if amount <= 0:
                  raise ValueError()
              Gateway().refund(self.id, amount)
              self.refunded_amount = amount
              return True
      """

  @domain
  Scenario: The exposed tool surface is tiny
    Then the agent tool surface is exactly "query, change, verify"
    And the agent tool surface costs at most 300 tokens

  @domain
  Scenario: A symbol query returns semantics without the source body
    Given the repository is compiled
    When the agent queries symbol "billing:Payment.refund"
    Then the answer includes "refund(self, amount: int) -> bool"
    And the answer includes "raises ValueError"
    And the answer does not include "self.refunded_amount = amount"
    And the answer costs fewer tokens than the source of "billing:Payment.refund"

  @integration
  Scenario: A structural change replaces one symbol and reports interface impact
    Given the repository is compiled
    When the agent changes "billing:Gateway.refund" to:
      """
      def refund(self, payment_id, amount):
          return payment_id is not None
      """
    Then the change is accepted
    And the change reports the interface as unchanged
    And file "billing.py" still defines "billing:Payment.refund"

  @integration
  Scenario: A change that does not define the target symbol is rejected
    Given the repository is compiled
    When the agent changes "billing:Gateway.refund" to:
      """
      def charge(self, payment_id, amount):
          return True
      """
    Then the change is rejected with "exactly one function named refund"
    And file "billing.py" is unmodified

  @integration
  Scenario: A change may add a decorator to a method
    Given the repository is compiled
    When the agent changes "billing:Payment.refund" to:
      """
      @functools.cache
      def refund(self, amount: int) -> bool:
          if amount <= 0:
              raise ValueError()
          Gateway().refund(self.id, amount)
          self.refunded_amount = amount
          return True
      """
    Then the change is accepted
    And the file "billing.py" includes the line "@functools.cache"

  @integration
  Scenario: A class-level change replaces the whole class
    Given the repository is compiled
    When the agent changes "billing:Gateway" to:
      """
      class Gateway:
          def refund(self, payment_id, amount):
              return payment_id is not None

          def charge(self, payment_id, amount):
              return True
      """
    Then the change is accepted
    And the change reports the interface as changed
    And file "billing.py" still defines "billing:Gateway.charge"

  @integration
  Scenario: A module-level constant is indexed and replaceable
    Given the repository is compiled
    When the agent queries symbol "billing:MAX_RETRIES"
    Then the answer includes "MAX_RETRIES = 3"
    When the agent changes "billing:MAX_RETRIES" to:
      """
      MAX_RETRIES = 5
      """
    Then the change is accepted
    And the change reports the interface as changed

  @integration
  Scenario: An unused constant can be removed through the ABI
    Given the repository is compiled
    When the agent removes "billing:MAX_RETRIES"
    Then the removal is accepted
    And the removal reports no remaining references
    And the symbol "billing:MAX_RETRIES" is gone from the IR

  @integration
  Scenario: Removing a symbol that is still called reports the dependents
    Given the repository is compiled
    When the agent removes "billing:Gateway.refund"
    Then the removal is accepted
    And the removal reports remaining references from "billing:Payment.refund"

  @integration
  Scenario: Verification returns a compact pass/fail result
    Given the repository is compiled
    When the agent verifies with a command that prints 50 lines and succeeds
    Then verification passed
    And the verification output has at most 20 lines

  @integration
  Scenario: The CLI compiles a repository into versioned IR JSON
    When I run the CLI with "compile {repo} --out {repo}/ir.json"
    Then the CLI exits with code 0
    And the file "ir.json" is IR version "1" containing symbol "billing:Payment.refund"

  @integration
  Scenario: The open build reserves slicing for the commercial optimizer
    When I run the CLI with "slice {repo} billing:Payment.refund"
    Then the CLI exits with code 3
    And the CLI output includes "commercial"
