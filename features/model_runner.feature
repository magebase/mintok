@integration
Feature: Model runner
  The benchmark harness builds Anthropic Messages API requests with the
  stdlib only. A fake transport stands in for the network: no scenario
  here performs a real HTTP call, and the API key never leaks into a
  request body.

  Scenario: The request targets the Messages endpoint with auth headers
    Given a fake Anthropic transport
    When the request is built for model "claude-sonnet-4-5" with api key "sk-test-0001"
    Then the request URL ends with "/v1/messages"
    And the request headers carry "x-api-key" and "anthropic-version"
    And the body carries the model, max_tokens, system and messages

  Scenario: The system prompt is marked for prompt caching
    Given a fake Anthropic transport
    When the request is built for model "claude-sonnet-4-5" with api key "sk-test-0001"
    Then the system prompt carries an ephemeral "cache_control" block

  Scenario: The api key never leaks into the request body
    Given a fake Anthropic transport
    When the request is built for model "claude-sonnet-4-5" with api key "sk-test-0001"
    Then the serialized body does not contain the api key

  Scenario: The reply usage maps onto the MinTok token convention
    Given an Anthropic reply with 1200 input, 8000 cache read, 2000 cache creation and 900 output tokens
    When the reply is parsed
    Then the usage record has 1200 input, 8000 cached input, 2000 cache write and 900 output tokens

  Scenario: A completion runs through a transport without touching the network
    Given a fake Anthropic transport
    And the model api key is set to "sk-test-0001"
    When a completion is run for system "You are a bench agent" and messages asking to reply "pong"
    Then the completion text is "pong"
    And the usage record has 1200 input, 8000 cached input, 2000 cache write and 900 output tokens

  Scenario: A missing api key fails before any network attempt
    Given the model api key is unset
    And a transport that must never be called
    When a completion is attempted
    Then a RuntimeError names the missing environment variable "MINTOK_MODEL_API_KEY"
