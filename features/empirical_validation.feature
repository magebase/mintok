Feature: Empirical Validation and Learned Policy Optimization

  As an AI agent architect
  I want an empirical, leakage-free evaluation harness and learned policy controller
  So that MinTok achieves proven Pareto-optimal inference efficiency on unseen repositories.

  @domain
  Scenario: Zero repository and task leakage partition
    Given a dataset of software engineering tasks across 6 distinct repositories
    When the leakage-free partitioner splits tasks into train, validation, and test holdout
    Then the train and test repositories must be strictly disjoint
    And no task in the training set may appear in the test holdout
    And the holdout test set contains tasks from at least 2 unseen repositories

  @domain
  Scenario: Multi-objective score J balances solve against regression and catastrophe
    Given a baseline policy with 80% solve rate, 15,000 tokens, 5% regression, and 1% catastrophe
    And a candidate policy with 90% solve rate, 7,200 tokens, 1% regression, and 0% catastrophe
    When the objective evaluator computes multi-objective score J
    Then the candidate policy J score must strictly exceed the baseline J score
    And a policy that aggressively aborts tasks to reduce tokens must suffer severe J penalty

  @domain
  Scenario: Dual-mode benchmark comparison
    Given a dual-mode benchmark configuration
    When evaluating policies under fixed solve target of 90 percent
    Then the learned policy achieves at least 80 percent token savings vs control
    And when evaluating policies under a fixed token budget of 10,000 tokens
    Then the learned policy achieves at least 40 percent absolute solve rate gain vs control

  @domain
  Scenario: Mixture of specialized policies routes actions to domain specialists
    Given a Mixture of Policies controller with 7 specialist policies
    When presented with a repository exceeding 100k LOC
    Then the controller routes to the LargeRepo specialist using AST slice
    And when presented with a failing test and zero patches
    Then the controller routes to the Debugging specialist
    And when presented with an unverified patch
    Then the controller routes to the Verification specialist
    And the average decision latency is strictly less than 1.0 milliseconds

  @domain
  Scenario: Frontier escalation auction chooses model tier via marginal ROI
    Given a Frontier Escalation Auction with a threshold of 30 probability gains per dollar
    When a local model has 85% solve probability and frontier has 88% solve probability for $0.075 cost
    Then the auction denies frontier escalation and keeps the local model
    And when local model has 20% solve probability and frontier has 85% solve probability
    Then the auction approves escalation with high marginal ROI

  @domain
  Scenario: Spend justification logs pre-action expectations and post-action yield
    Given a spend justification logger
    When a pre-action justification is recorded for a slice action with expected solve gain 0.20
    And the post-action outcome achieves solve gain 0.22 with 1,200 tokens spent
    Then the action is marked as justified with positive ROI

  @domain
  Scenario: Adversarial benchmark suite tests 8 failure modes
    Given the 8-case adversarial benchmark suite
    When the benchmark suite is executed
    Then MinTok passes all 8 adversarial stress cases
    And MinTok prevents the cheap-looking catastrophic failure
    And MinTok achieves at least 80 percent token savings across the adversarial suite

  @integration
  Scenario: CLI eval-holdout runs leakage-free holdout evaluation
    When the user runs mintok eval-holdout with json format
    Then the CLI exits with code 0
    And the output contains zero data leakage
    And the output reports dual-mode evaluation metrics

  @integration
  Scenario: CLI adversarial executes 8-case stress suite
    When the user runs mintok adversarial with json format
    Then the CLI exits with code 0
    And all 8 adversarial cases report passed
    And at least 1 catastrophe is prevented

  @integration
  Scenario: CLI simulate executes Monte Carlo trajectory simulation
    When the user runs mintok simulate with 1000 trajectories
    Then the CLI exits with code 0
    And the simulated solve rate exceeds 85 percent
    And the controller latency is below 1.0 milliseconds
