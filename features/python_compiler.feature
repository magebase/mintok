@domain
Feature: Python semantic compiler
  The compiler turns Python source into Agent Program IR so agents query
  verifiable semantic facts instead of reading raw source text.

  Background:
    Given a repository file "billing.py":
      """
      class AlreadyRefunded(Exception):
          pass


      class Gateway:
          def refund(self, payment_id, amount):
              return True


      class Payment:
          def refund(self, amount: int) -> bool:
              if self.refunded_amount:
                  raise AlreadyRefunded()
              Gateway().refund(self.id, amount)
              self.refunded_amount = amount
              return True
      """

  Scenario: Functions and methods become symbols with signatures
    When the repository is compiled
    Then the IR contains symbol "billing:Payment.refund" of kind "method"
    And the IR contains symbol "billing:Gateway" of kind "class"
    And symbol "billing:Payment.refund" has signature "refund(self, amount: int) -> bool"

  Scenario: Effects are extracted as facts
    When the repository is compiled
    Then symbol "billing:Payment.refund" has fact "raises" "AlreadyRefunded"
    And symbol "billing:Payment.refund" has fact "writes" "billing:Payment.refunded_amount"
    And symbol "billing:Payment.refund" calls "billing:Gateway.refund" with confidence 1.00

  Scenario: Callers are answered by the compiler, not the model
    Given a repository file "api.py":
      """
      from billing import Payment


      def refund_endpoint(payment: Payment, amount: int):
          return payment.refund(amount)
      """
    When the repository is compiled
    Then the callers of "billing:Payment.refund" are "api:refund_endpoint"

  Scenario: Formatting, comment, and docstring changes keep every hash
    When the repository is compiled
    And file "billing.py" is replaced with:
      """
      class AlreadyRefunded(Exception):
          pass


      class Gateway:
          def refund(self, payment_id, amount):
              '''Refund through the payment gateway.'''
              # Always succeeds in the sandbox gateway.
              return (True)


      class Payment:
          def refund(self, amount: int) -> bool:
              if self.refunded_amount:
                  raise AlreadyRefunded()
              Gateway().refund(
                  self.id,
                  amount,
              )
              self.refunded_amount = amount
              return True
      """
    And the repository is compiled again
    Then the body hash of "billing:Gateway.refund" is unchanged
    And the interface hash of "billing:Payment.refund" is unchanged

  Scenario: A body change that keeps the interface does not force relearning
    When the repository is compiled
    And file "billing.py" is replaced with:
      """
      class AlreadyRefunded(Exception):
          pass


      class Gateway:
          def refund(self, payment_id, amount):
              return payment_id is not None


      class Payment:
          def refund(self, amount: int) -> bool:
              if self.refunded_amount:
                  raise AlreadyRefunded()
              Gateway().refund(self.id, amount)
              self.refunded_amount = amount
              return True
      """
    And the repository is compiled again
    Then the body hash of "billing:Gateway.refund" is changed
    And the interface hash of "billing:Gateway.refund" is unchanged

  Scenario: A new effect changes the interface hash
    When the repository is compiled
    And file "billing.py" is replaced with:
      """
      class AlreadyRefunded(Exception):
          pass


      class Gateway:
          def refund(self, payment_id, amount):
              return True


      class Payment:
          def refund(self, amount: int) -> bool:
              if self.refunded_amount:
                  raise AlreadyRefunded()
              if amount <= 0:
                  raise ValueError()
              Gateway().refund(self.id, amount)
              self.refunded_amount = amount
              return True
      """
    And the repository is compiled again
    Then the interface hash of "billing:Payment.refund" is changed
