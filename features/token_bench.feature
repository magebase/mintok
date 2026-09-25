@integration
Feature: Context-token benchmark
  The deterministic benchmark pairs representative agent questions against a
  grep-and-paging baseline and reports the measured context-token reduction.
  It needs no model calls: both arms construct the context that answers each
  question, and tokens are estimated with the project estimator.

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
    And a repository file "notify.py":
      """
      def send_email(address, message):
          if not address:
              raise ValueError()
          return True


      def notify_all(addresses, message):
          results = []
          for address in addresses:
              results.append(send_email(address, message))
          return results


      class Notifier:
          def broadcast(self, addresses, message):
              self.sent = notify_all(addresses, message)
              return self.sent
      """

  @integration
  Scenario: The benchmark measures a reduction on every task
    When the token benchmark runs alone on "{repo}" with JSON output
    Then the CLI exits with code 0
    And the benchmark JSON reports baseline tokens above treatment tokens
    And every benchmark task costs fewer treatment tokens than baseline tokens
    And the benchmark JSON reports a reduction ratio of at least "1.5"
    And the benchmark JSON reports a strong-tooling reduction ratio of at least "1.0"

  @integration
  Scenario: The text report states the reduction
    When the token benchmark runs alone on "{repo}" as text
    Then the CLI exits with code 0
    And the CLI output includes "token reduction"

  @integration
  Scenario: The benchmark includes relearning against a second version
    Given a copy of the repository is made
    And file "billing.py" in the copy is replaced with:
      """
      class Gateway:
          def refund(self, payment_id, amount):
              return payment_id is not None


      class Payment:
          def refund(self, amount: float) -> bool:
              '''Refund part or all of a captured payment.'''
              if amount <= 0:
                  raise ValueError()
              Gateway().refund(self.id, amount)
              self.refunded_amount = amount
              return True
      """
    When the token benchmark runs with a second version on "{repo}" against "{copy}" with JSON output
    Then the CLI exits with code 0
    And the benchmark JSON includes a "relearn" task
