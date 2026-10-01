@domain
Feature: Fast Iteration Loop and Counterfactual Policy Evaluation
  To optimize developer velocity without losing scientific signal,
  MinTok provides a four-tier evaluation ladder, offline counterfactual
  trajectory replay, a fixed 12-canary suite, and a learned local policy predictor.

  @domain
  Scenario: Tier 0 mechanism smoke test enforces the dev promotion gate
    Given candidate policy "v3" and control policy "control"
    When the Tier 0 mechanism smoke test runs on 8 fixed tasks
    Then 8 tasks are evaluated across small edits, cross-file, test failures, large modules, ambiguous, and easy control
    And the smoke report records provider tokens, frontier turns, and verification success
    And the delta solve is at least "-1"
    And the tokens per solved ratio is at most "0.80"
    And zero catastrophic regressions are detected
    And the dev promotion gate verdict is "PASS"

  @domain
  Scenario: Tier 0 smoke test rejects candidates with catastrophic regressions
    Given candidate policy "v3_broken" with a forced catastrophic failure
    When the Tier 0 mechanism smoke test runs on 8 fixed tasks
    Then the dev promotion gate verdict is "REJECT"
    And the rejection reason mentions "Catastrophic regression detected"

  @domain
  Scenario: Fixed canary suite defines 12 critical failure modes
    Given the MinTok canary suite configuration
    Then MINTOK_CANARY equals 12
    And canary tasks cover large-module localization, class target, cross-file, failing test, and misleading traceback

  @domain
  Scenario: Tier 1 paired mini-benchmark dev-20 interleaves task execution
    Given benchmark suite "dev-20" with arms "control,v3"
    When the paired mini-benchmark executes with interleaving enabled
    Then exactly 20 tasks are evaluated across stratified task families
    And execution order alternates between "control -> v3" and "v3 -> control"
    And the provider tokens ratio is at most "0.40"
    And the candidate solves at least 18 tasks out of 20
    And the report includes tokens per solved, p95 tokens, and verification counts

  @domain
  Scenario: Offline counterfactual trajectory replay analyzes savings and evidence loss
    Given historical trajectory records from previous runs
    When the counterfactual replay engine simulates policy "v3_candidate"
    Then the context token reduction ratio is at least "1.5"
    And recovered observations exceed 70 percent
    And predicted lost evidence is below 5 percent
    And tasks are partitioned into definitely unaffected and potentially affected sets

  @domain
  Scenario: Learned local policy predictor ranks actions by net utility
    Given an agent state with active failures and high token consumption
    And candidate actions including "read_full", "read_slice", "run_verifier", and "compact_state"
    When the policy predictor scores candidate actions with solve value 1.0 and token lambda 0.000005
    Then every candidate action receives a P(solve), expected tokens, and utility score
    And "read_slice" or "run_verifier" achieves higher utility than "read_full"

  @domain
  Scenario: Content-addressed caching guarantees zero inference for identical experiment setups
    Given an inference setup with model, system prompt, task, repo snapshot, and tools
    When the content-addressed cache key is computed
    Then the key matches the SHA256 digest of the complete configuration tuple
    And querying the cache with identical configuration requires zero model calls

  @domain
  Scenario: Frozen 12-task fast set stratifies tasks across 6 categories
    Given the frozen 12-task fast set specification
    Then exactly 12 tasks are included
    And the set contains 2 small localized tasks, 2 cross-file tasks, 2 large-module tasks, 2 test debugging tasks, 2 API schema tasks, and 2 difficult failure cases

  @domain
  Scenario: Three checker tiers control evaluation turn cap and scope
    Given the three checker tiers "FAST", "DEV", and "RELEASE"
    Then "FAST" specifies 12 tasks with a turn cap of 12
    And "DEV" specifies 25 tasks with full checker
    And "RELEASE" specifies up to 200 tasks with frozen environment and full accounting

  @domain
  Scenario: Stop-loss controller enforces evidence gathering after 3 failed patches
    Given a stop-loss controller tracking patch attempts
    When 3 consecutive patches fail without evidence gathering
    Then a proposed "apply_patch" action is intercepted with verdict "FORCE_EVIDENCE_GATHERING"
    And the controller demands investigation or testing before next edit

  @domain
  Scenario: Stop-loss controller blocks expanding the same observation more than twice
    Given a stop-loss controller tracking observation handles
    When observation "obs_traceback_42" is expanded 2 times
    Then a proposed "expand" action on "obs_traceback_42" is intercepted with verdict "BLOCK_EXPANSION_REDIRECT"

  @domain
  Scenario: Evidence sufficiency model locks reads when all 5 gates are satisfied
    Given evidence sufficiency state with target, caller, failure, patch location, and expected behavior known
    When the agent proposes an expensive "read_full" action
    Then the MinTok controller intercepts the read and directs the agent to "apply_patch"

  @domain
  Scenario: Adaptive verifier scales verification scope by change blast radius
    Given the adaptive verifier
    When blast radius is evaluated for a local function change
    Then the verifier selects "unit_test" scope with an estimated token budget below 1000
    When blast radius is evaluated for a public API signature modification
    Then the verifier selects "affected_tests" scope

  @domain
  Scenario: Local action critic flags unverified patches before another edit
    Given the local action critic and a state with a pending unverified patch
    When the agent proposes another "apply_patch" action
    Then the critic classifies the action as "NEEDS_VERIFICATION"

  @domain
  Scenario: Champion versus Challenger tournament promotes Pareto improvements
    Given incumbent champion with 10 solves and 300000 tokens on the fast set
    When challenger policy achieves 10 solves and 210000 tokens
    Then the tournament verdict is promoted with Pareto improvement confirmed
    When another challenger achieves 8 solves and 150000 tokens
    Then the tournament verdict is rejected due to solve regression

  @domain
  Scenario: Hypothesis graph tracks hierarchical tree and updates posterior confidence
    Given a hypothesis graph with root "Tax calculation bug" and child "VAT rate exemption logic"
    When evidence is observed that VAT exemptions are disabled in config
    Then the child hypothesis confidence updates and Shannon entropy is reduced

  @domain
  Scenario: Negative evidence memory prevents redundant exploration of dead ends
    Given negative evidence memory with verified fact "NO_CALLER_WRITES_TIMEOUT" for target "billing/service.py"
    When the agent proposes reading or querying "billing/service.py" for timeout writes
    Then the MinTok controller intercepts the action citing the negative evidence memory

  @domain
  Scenario: Observation dependency graph performs surgical line-range invalidation
    Given a cached observation for "billing/calc.py" covering lines 100 to 150
    And another cached observation covering lines 10 to 30
    When a patch hunk modifies "billing/calc.py" from line 15 to line 25
    Then the observation covering lines 10 to 30 is invalidated
    And the observation covering lines 100 to 150 is retained with its context tokens intact

  @domain
  Scenario: Observation cache enforces tiered TTL lifecycle
    Given observations with TTL modes "RUN", "UNTIL_PATCH", and "IMMEDIATE"
    When a turn advances, "IMMEDIATE" observations are expired
    And a code patch is applied, "UNTIL_PATCH" observations are invalidated
    Then "RUN" observations remain valid throughout the entire session

  @domain
  Scenario: Action-value model evaluates downstream observation value over raw action cost
    Given an action value model and candidate actions "read_caller" and "read_full"
    When the observation value is evaluated
    Then "read_caller" yields higher net observation value than "read_full" by saving future search tokens

  @domain
  Scenario: Propensity logging records action distributions and rejected alternatives for OPE
    Given the MinTok controller processing agent steps with active exploration
    When decisions are evaluated across candidate actions
    Then every decision records an action propensity distribution summing to 1.0
    And the chosen action and all rejected alternatives are logged with their utility scores

  @domain
  Scenario: Off-policy evaluation estimates candidate policy value via importance sampling
    Given logged trajectory propensity records from a historical run
    When off-policy evaluation simulates a candidate context allocation policy
    Then importance sampling and weighted importance sampling rewards are computed with effective sample size

  @domain
  Scenario: Survival predictor triggers early strategy pivots before catastrophic runaway
    Given an agent trajectory with 3 failed patches, 45000 tokens spent, and repeated actions
    When survival analysis evaluates the trajectory state
    Then the runaway hazard exceeds 0.70 and the recommendation is "EARLY_ABORT" or "PIVOT_STRATEGY"

  @domain
  Scenario: Trajectory fingerprinter classifies behavioral archetypes
    Given an agent trajectory with 4 consecutive reads without editing and 20000 tokens spent
    When trajectory fingerprinting classifies the behavior
    Then the archetype is identified as "OVER_READER" with an intervention recommending token budgeting

  @domain
  Scenario: Pre-action token budgeting and progressive precision ladder adapt tool envelopes
    Given a pre-action token budget of 600 tokens
    When the budget negotiator formats a "read_slice" action
    Then the action is configured with "compact_slice" format and CALL_GRAPH precision within budget

  @domain
  Scenario: Prompt-cache economics models cache hit ratio and prefix stability
    Given existing cached context of 20000 tokens
    When comparing appending 2000 tokens versus rewriting the entire context
    Then appending preserves the prompt-cache prefix and costs substantially less than rewriting

  @domain
  Scenario: Trajectory state machine transitions through phases and triggers ACT_NOW
    Given a trajectory state machine in phase "DISCOVERY"
    When target is known, fault is localized, and expected behavior is confirmed
    Then the state machine transitions to phase "PATCHING"
    And ACT_NOW value of waiting confirms the agent should execute immediately

  @domain
  Scenario: Tool complementarity model scores synergistic macro-action plans
    Given a macro-planner generating sequences for phase "LOCALIZATION"
    When a macro-plan is generated with steps "query_symbol", "query_callers", and "read_slice"
    Then the plan contains exactly 3 steps within budget
    And the tool complementarity score is strictly positive due to pairwise synergy

  @domain
  Scenario: Minimum sufficient patch predictor detects hypothesis loss and patch bloat
    Given a bugfix task expecting a 1-file 8-line modification
    When an agent proposes a bloated patch touching 4 files and 180 lines
    Then the patch anomaly verdict is "REASSESS_HYPOTHESIS" with bloat ratio exceeding 10.0

  @domain
  Scenario: Token arbitrage layer rejects frontier escalation when marginal intelligence is low
    Given model options "cheap_api" with 0.79 solve at $0.015 and "frontier" with 0.80 solve at $0.060
    When marginal intelligence escalation is evaluated
    Then the escalation to frontier is rejected because +0.01 solve does not justify the 4x cost delta
    And "cheap_api" is selected as the winning execution tier

  @domain
  Scenario: Dynamic state-dependent oracle and avoidable spend accounting measure efficiency
    Given an agent trajectory spending 40000 tokens on a 50-line target
    When the dynamic oracle and avoidable spend breakdown are computed
    Then the dynamic oracle specifies less than 1000 tokens
    And avoidable spend categorizes redundancy, recovery, and catastrophic waste


