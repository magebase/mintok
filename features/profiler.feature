@domain
Feature: Inference profiler
  The free profiler reads agent session records and quantifies avoidable
  inference spend. It optimizes nothing; it makes waste visible, with
  published formulas, as the first stage of the open-core funnel.

  Scenario: Total spend and avoidable waste by category
    Given these agent session records:
      | session | task | class         | model         | input_tokens | output_tokens | cache_read_tokens | usd   | context_reads    |
      | s1      | t1   | schema_change | opus-frontier | 36000        | 4000          | 30000             | 40.00 | billing.py:4000:4 |
      | s2      | t2   | deterministic | opus-frontier | 20000        | 5000          | 20000             | 30.00 | api.py:1000:1    |
      | s3      | t3   | bugfix        | opus-frontier | 10000        | 4200          | 1000              | 14.20 | util.py:2000:2   |
    When the profile is computed
    Then total spend is "$84.20"
    And avoidable "repeated context" is "$14.00"
    And avoidable "overpowered model" is "$30.00"
    And avoidable "poor cache use" is "$3.55"
    And estimated avoidable is "$47.55"

  Scenario: The improvement multiple frames the optimization opportunity
    Given these agent session records:
      | session | task | class         | model         | input_tokens | output_tokens | cache_read_tokens | usd   | context_reads    |
      | s1      | t1   | schema_change | opus-frontier | 36000        | 4000          | 30000             | 40.00 | billing.py:4000:4 |
      | s2      | t2   | deterministic | opus-frontier | 20000        | 5000          | 20000             | 30.00 | api.py:1000:1    |
      | s3      | t3   | bugfix        | opus-frontier | 10000        | 4200          | 1000              | 14.20 | util.py:2000:2   |
    When the profile is computed
    Then the potential improvement is "2.30x"

  Scenario: Waste categories have documented formulas
    Then "repeated context" prices every read beyond the first at the session's average cost per token
    And "overpowered model" flags the full cost of frontier sessions on deterministic task classes
    And "poor cache use" flags 25% of cost when fewer than half the input tokens hit cache

  Scenario: Overlapping waste never claims more than total spend
    Given these agent session records:
      | session | task | class         | model         | input_tokens | output_tokens | cache_read_tokens | usd  | context_reads |
      | s1      | t1   | deterministic | opus-frontier | 10000        | 2000          | 10000             | 5.00 | a.py:1000:3  |
    When the profile is computed
    Then total spend is "$5.00"
    And estimated avoidable is "$5.00"
    And the potential improvement is unbounded

  @integration
  Scenario: The CLI profiles a session JSONL file
    Given a session JSONL file with $84.20 total spend
    When I run the CLI with "profile {repo}/sessions.jsonl"
    Then the CLI exits with code 0
    And the CLI output includes "$84.20"
    And the CLI output includes "2.30x"
