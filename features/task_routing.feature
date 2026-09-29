@domain
Feature: Deterministic pre-flight task router
  Routing is the first efficiency lever the frozen eval justified: task
  class alone captures 1.39x over uniform control. The router must decide
  using only information available before the frontier model starts —
  instruction wording and target file sizes — and must log every prediction
  against the eventual actual outcome so a future learned router can be
  trained on the accumulated dataset. Deterministic until rules stop
  improving.

  Scenario: A signature-propagation task routes to the semantic backend
    Given an instruction "Add a tip_cents keyword parameter (default 0) to compute_total in src/shopcart/pricing.py"
    And target files: "src/shopcart/pricing.py"=118
    When the task is routed
    Then the backend is "semantic-C"
    And the predicted class is "api_signature_propagation"
    And the expected relative cost is below 1.0
    And the reasons include "keyword parameter"

  Scenario: A rename-with-callers refactor routes to control even on a small file
    Given an instruction "Rename OrderService.place_order to checkout in src/shopcart/orders.py and update every caller (including tests) so the fixture suite passes"
    And target files: "src/shopcart/orders.py"=95
    When the task is routed
    Then the backend is "control"
    And the predicted class is "refactor"
    And the reasons include "rename"

  Scenario: A large-module behavior fix routes to the slicer backend
    Given an instruction "In src/biglib/large.py, locate count_index_tokens and make it ignore the common stopwords 'the', 'a' and 'of' when counting"
    And target files: "src/biglib/large.py"=1363
    When the task is routed
    Then the backend is "slicer"
    And the predicted class is "large_file_navigation"

  Scenario: A cross-file instruction routes to control
    Given an instruction "OrderService.order_summary in src/shopcart/orders.py misuses pricing.compute_total in src/shopcart/pricing.py by passing order.total_cents as the subtotal"
    And target files: "src/shopcart/orders.py"=95, "src/shopcart/pricing.py"=61
    When the task is routed
    Then the backend is "control"
    And the predicted class is "cross_file_bug"

  Scenario: A schema task on the large module still routes to the semantic backend
    Given an instruction "Add a revision field (default 1) to the Entry dataclass in src/biglib/large.py and update the module index writer"
    And target files: "src/biglib/large.py"=1363
    When the task is routed
    Then the backend is "semantic-C"
    And the predicted class is "schema_or_framework_change"

  Scenario: An instruction with no cues defaults to the semantic backend at low confidence
    Given an instruction "Make parse_money in src/shopcart/currency.py handle the empty input gracefully"
    And target files: "src/shopcart/currency.py"=48
    When the task is routed
    Then the backend is "semantic-C"
    And the confidence is below 0.75

  Scenario: Every prediction is logged as features, choice, and outcome
    Given an instruction "Add a tip_cents keyword parameter (default 0) to compute_total in src/shopcart/pricing.py"
    And target files: "src/shopcart/pricing.py"=118
    When the task "shopcart-api-01" is routed with actual outcomes control=2500/solved and semantic-C=1200/solved
    Then the record selects "semantic-C"
    And the record contains the feature "target_loc" with value 118
    And the record contains "actual" for both backends

  Scenario: High-complexity monorepo profile routes to control
    Given an instruction "Add a tip_cents keyword parameter (default 0) to compute_total in src/azure/pricing.py"
    And target files: "src/azure/pricing.py"=118
    And a repository profile with complexity 0.75 and strategy "virtualized-shell"
    When the task is routed
    Then the backend is "control"
    And the predicted class is "monorepo_complexity"
    And the reasons include "monorepo complexity"

  Scenario: Prediction record calculates utility for paired comparison
    Given an instruction "Fix token count in src/shopcart/calc.py"
    And target files: "src/shopcart/calc.py"=50
    When the task "utility-task-01" is routed with actual outcomes control=2000/solved and semantic-C=1000/solved
    Then the utility for "semantic-C" is higher than "control"
