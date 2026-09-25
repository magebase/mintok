@domain
Feature: Compact context queries
  Deterministic query ops answer "where is it", "who writes this", and "what
  does it do" from the semantic IR alone, so agents never page through raw
  source to locate or classify code.

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
  Scenario: A find query answers with one signature line per match
    Given the repository is compiled
    When the agent finds symbols matching "refund"
    Then the find answer lists symbols "billing:Gateway.refund, billing:Payment.refund"
    And the find answer costs fewer tokens than the source of "billing:Payment.refund"

  @domain
  Scenario: A writers query locates attribute writers without source
    Given the repository is compiled
    When the agent queries writers of attribute "refunded_amount"
    Then the answer names writer "billing:Payment.refund writes billing:Payment.refunded_amount"

  @domain
  Scenario: A summary query adds the docstring line without the body
    Given the repository is compiled
    When the agent queries summary "billing:Payment.refund"
    Then the summary answer includes "Refund part or all of a captured payment."
    And the summary answer includes "raises ValueError"
    And the summary answer omits "self.refunded_amount = amount"
