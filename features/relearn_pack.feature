@domain
Feature: Relearn packs
  After a change, agents relearn only what interface hashes force: added,
  removed, and interface-changed symbols arrive as compact facts, while
  body-only changes are provably free to skip and are omitted from the pack.

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
    And the repository is compiled

  @domain
  Scenario: A body-only change is omitted from the pack
    When file "billing.py" is replaced with:
      """
      class Gateway:
          def refund(self, payment_id, amount):
              return payment_id is not None


      class Payment:
          def refund(self, amount: int) -> bool:
              '''Refund part or all of a captured payment.'''
              if amount <= 0:
                  raise ValueError()
              Gateway().refund(self.id, amount)
              self.refunded_amount = amount
              return True
      """
    And the repository is compiled again
    And the relearn pack is built
    Then the pack omits "billing:Gateway.refund"
    And the pack reports "1 body-only change(s) omitted"

  @domain
  Scenario: An interface change is included with its new facts
    When file "billing.py" is replaced with:
      """
      class Gateway:
          def refund(self, payment_id, amount):
              return True


      class Payment:
          def refund(self, amount: float) -> bool:
              '''Refund part or all of a captured payment.'''
              if amount <= 0:
                  raise ValueError()
              Gateway().refund(self.id, amount)
              self.refunded_amount = amount
              return True
      """
    And the repository is compiled again
    And the relearn pack is built
    Then the pack includes "interface billing:Payment.refund"
    And the pack rendering of "billing:Payment.refund" includes "refund(self, amount: float) -> bool"
    And the pack costs fewer tokens than the source file "billing.py"
