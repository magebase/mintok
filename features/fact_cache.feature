@domain
Feature: Hash-invalidated fact cache
  Cached facts are mechanically verifiable and tied to the source hash they
  were derived from, so edits invalidate only what actually changed.

  Background:
    Given a repository file "billing.py":
      """
      class Gateway:
          def refund(self, payment_id, amount):
              return True


      class Payment:
          def refund(self, amount: int) -> bool:
              Gateway().refund(self.id, amount)
              return True
      """
    And the repository is compiled
    And the compiled facts are cached

  Scenario: Only facts for the edited symbol are invalidated
    When file "billing.py" is replaced with:
      """
      class Gateway:
          def refund(self, payment_id, amount):
              return True


      class Payment:
          def refund(self, amount: int) -> bool:
              if amount <= 0:
                  raise ValueError()
              Gateway().refund(self.id, amount)
              return True
      """
    And the repository is compiled again
    And the cache is refreshed from the new compile
    Then the invalidated symbols are exactly "billing:Payment.refund"
    And cached facts for "billing:Gateway.refund" are still valid

  Scenario: A formatting-only edit invalidates nothing
    When file "billing.py" is replaced with:
      """
      class Gateway:
          def refund(self, payment_id, amount):
              # sandbox
              return True


      class Payment:
          def refund(self, amount: int) -> bool:
              Gateway().refund(self.id,
                               amount)
              return True
      """
    And the repository is compiled again
    And the cache is refreshed from the new compile
    Then no symbols are invalidated

  Scenario: A deleted symbol drops its facts
    When file "billing.py" is replaced with:
      """
      class Gateway:
          def refund(self, payment_id, amount):
              return True
      """
    And the repository is compiled again
    And the cache is refreshed from the new compile
    Then the invalidated symbols are exactly "billing:Payment, billing:Payment.refund"
    And the cache holds no facts for "billing:Payment.refund"
