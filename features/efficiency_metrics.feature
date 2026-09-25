@domain
Feature: Efficiency metrics and experiment discipline
  The score is accepted changes per total dollar, not tokens. Claims need
  paired evaluation and confidence intervals because runs are noisy.

  Scenario: Efficiency counts every cost component
    Given these benchmark runs:
      | task | arm      | solved | frontier_usd | local_usd | indexing_usd | frontier_calls |
      | t1   | baseline | yes    | 2.00         | 0.00      | 0.00         | 8              |
      | t2   | baseline | no     | 2.00         | 0.00      | 0.00         | 10             |
      | t1   | mintok   | yes    | 0.40         | 0.05      | 0.05         | 2              |
      | t2   | mintok   | yes    | 0.40         | 0.05      | 0.05         | 2              |
    Then arm "baseline" achieves 0.25 accepted changes per dollar
    And arm "mintok" achieves 2.00 accepted changes per dollar
    And arm "baseline" uses 18.0 frontier calls per success
    And arm "mintok" uses 2.0 frontier calls per success
    And the efficiency ratio of "mintok" over "baseline" is 8.00

  Scenario: Fewer tokens at a higher total cost is a regression
    Given these benchmark runs:
      | task | arm        | solved | frontier_usd | local_usd | indexing_usd | frontier_calls |
      | t1   | baseline   | yes    | 1.00         | 0.00      | 0.00         | 4              |
      | t1   | compressed | yes    | 1.44         | 0.00      | 0.00         | 7              |
    Then the efficiency ratio of "compressed" over "baseline" is 0.69

  Scenario: A large paired improvement is significant
    Given 30 paired tasks where "baseline" costs 2.00 and "mintok" costs 0.50 per task
    And both arms solve tasks 1 to 20
    When a paired bootstrap of the efficiency ratio is run with seed 7
    Then the ratio confidence interval lower bound is above 3.0
    And the result is "significant"

  Scenario: A small difference inside the noise floor is not significant
    Given 30 paired tasks where "baseline" costs 1.00 and "mintok" costs 1.00 per task
    And "baseline" solves tasks 1 to 15 and "mintok" solves tasks 2 to 17
    When a paired bootstrap of the efficiency ratio is run with seed 7
    Then the result is "not significant"

  Scenario Outline: The oracle experiment gates further investment
    Given the oracle-context arm is <headroom>x more efficient than the baseline
    Then the oracle verdict is "<verdict>"

    Examples:
      | headroom | verdict     |
      | 1.5      | KILL        |
      | 2.4      | INVESTIGATE |
      | 6.0      | CONTINUE    |
