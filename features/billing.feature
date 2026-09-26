@domain
Feature: API billing instrumentation
  Usage records and a price table turn raw token counts into itemized,
  verifiable dollar costs. By convention input_tokens excludes cached
  tokens: cache hits are billed at the cached_input rate and cache writes
  at the cache_write rate.

  Background:
    Given the default price table

  Scenario: A cache-heavy turn costs far less than the same volume uncached
    Given claude-sonnet-4-5 usage with 10000 input, 90000 cached input, 0 cache write, 1000 output and 0 reasoning tokens
    When the cost is computed
    Then the itemized "input" cost is 0.03
    And the itemized "cached_input" cost is 0.027
    And the total cost is 0.072
    When the same token volume is billed with no cache hits
    Then the total cost is 0.315
    And the cached turn is cheaper than the uncached turn

  Scenario: Cache writes carry a premium over fresh input
    Given claude-sonnet-4-5 usage with 5000 input, 0 cached input, 10000 cache write, 500 output and 0 reasoning tokens
    When the cost is computed
    Then the itemized "input" cost is 0.015
    And the itemized "cache_write" cost is 0.0375
    And the itemized "output" cost is 0.0075
    And the total cost is 0.06

  Scenario: An unknown model is rejected with the known models listed
    Given claude-sonnet-4-5 usage with 1000 input, 0 cached input, 0 cache write, 100 output and 0 reasoning tokens
    When the cost is computed for unknown model "gpt-4o-mini"
    Then a KeyError names the unknown model and lists "claude-sonnet-4-5" and "gpt-5"

  Scenario: Negative token counts are rejected
    When a usage record is built with -1 input tokens
    Then a ValueError names the offending field

  Scenario: A billing row round-trips through JSON
    Given claude-sonnet-4-5 usage with 10000 input, 90000 cached input, 0 cache write, 1000 output and 0 reasoning tokens
    And its cost is computed
    When a billing row is emitted for run "r1", task "t1", turn 3
    And the row is serialized and parsed back through JSON
    Then the row carries the tokens, the itemized costs and the total
