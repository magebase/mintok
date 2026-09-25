Feature: Agent ABI
  Frontier agents learn one compact interface over the semantic IR instead of
  dozens of verbose tools and raw source reads.

  Background:
    Given a repository file "billing.py":
      """
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
