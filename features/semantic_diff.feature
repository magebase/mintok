@domain
Feature: Semantic diff
  Reviewers and agents see meaning-level changes instead of line diffs:
  interface changes force relearning, body-only changes do not, and symbols
  that did not change are not reported at all.

  Background:
    Given a repository "old" containing file "billing.py":
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
    And a repository "new" containing file "billing.py":
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
              if amount <= 0:
                  raise ValueError()
              Gateway().refund(self.id, amount)
              self.refunded_amount = amount
              return True


      class Ledger:
          def commit(self):
              return True
      """
    And both repositories are compiled

  Scenario: Interface change, body-only change, and addition are distinguished
    When the semantic diff is computed
    Then the diff reports "interface" for "billing:Payment.refund"
    And the diff detail for "billing:Payment.refund" includes "+raises ValueError"
    And the diff reports "body" for "billing:Gateway.refund"
    And the diff reports "added" for "billing:Ledger"
    And the diff reports "added" for "billing:Ledger.commit"
    And the diff does not report "billing:AlreadyRefunded"

  @integration
  Scenario: The CLI prints a semantic diff between two roots
    When I run the CLI with "diff {repo}/old {repo}/new"
    Then the CLI exits with code 0
    And the CLI output includes "interface billing:Payment.refund"
    And the CLI output includes "billing:Ledger.commit"

  @integration
  Scenario: The CLI can emit the diff as JSON
    When I run the CLI with "diff {repo}/old {repo}/new --format json"
    Then the CLI exits with code 0
    And the CLI output includes "kind"
    And the CLI output includes "billing:Payment.refund"
