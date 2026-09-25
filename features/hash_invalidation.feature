@domain
Feature: Hash invalidation correctness
  The two-hash design is only safe if it changes exactly when meaning-relevant
  knowledge changes: presentation-only edits must not invalidate, while every
  new externally visible effect must. A hash that fails to invalidate silently
  destroys correctness; one that over-invalidates destroys the savings.

  Background:
    Given a repository file "billing.py":
      """
      class Gateway:
          def refund(self, payment_id, amount):
              return True


      class Payment:
          def audit(self):
              return True

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
  Scenario: Invalidation matches the meaning of every mutation
    When every named mutation is applied and recompiled
    Then invalidation matches expectation:
      | mutation           | body_changes | interface_changes |
      | comment            | no           | no                |
      | docstring          | no           | no                |
      | formatting         | no           | no                |
      | rename_local       | yes          | no                |
      | reorder_condition  | yes          | no                |
      | add_print          | yes          | no                |
      | new_exception      | yes          | yes               |
      | change_return_type | yes          | yes               |
      | new_call           | yes          | yes               |
      | new_write          | yes          | yes               |
      | retarget_write     | yes          | yes               |
