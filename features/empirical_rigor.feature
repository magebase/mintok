Feature: Empirical Rigor, Simulator Calibration, and Cross-Model Generalization

  As an AI research engineer
  I want rigorous calibration, expanded statistical adversarial testing, and cross-model frontier evaluation
  So that MinTok's performance claims are supported by calibrated, leakage-free empirical proof.

  @domain
  Scenario: Simulator calibration validates predictions against real holdout trajectories
    Given a holdout set of real model execution trajectories
    When the simulator calibrator evaluates predicted probabilities against actual outcomes
    Then the Expected Calibration Error is strictly below 8.0 percent
    And the Brier score is strictly below 0.15
    And the Token Prediction MAPE is strictly below 20.0 percent
    And the simulator is officially marked as qualified to run counterfactual rollouts

  @domain
  Scenario: Expanded adversarial benchmark evaluates 10 families across 50 tasks
    Given an expanded adversarial benchmark suite of 50 tasks across 10 failure mode families
    When the benchmark suite executes against 5 distinct repositories
    Then MinTok achieves an overall solve rate of at least 95 percent
    And Control achieves an overall solve rate below 25 percent
    And MinTok prevents all catastrophic failure cases
    And the performance difference is statistically significant with p less than 0.001

  @domain
  Scenario: 12-Arm ablation ladder isolates the marginal contribution of every mechanism
    Given the 12-arm ablation ladder from Arm A Control through Arm L Full MinTok
    When all 12 arms are evaluated on identical frozen benchmark tasks
    Then every successive arm from A to L maintains or increases the multi-objective J score
    And Virtualization and State Compilation account for more than 50 percent of total token savings
    And the Full MinTok policy achieves over 80 percent token reduction vs Control

  @domain
  Scenario: Three-scoreboard generalization harness verifies frozen repository disjointness
    Given a dataset partitioned into 70% Train, 15% Validation, and 15% Final Holdout by repository
    When the generalization harness evaluates the three scoreboards
    Then Scoreboard 1 confirms 100 percent engineering specification compliance
    And Scoreboard 2 confirms controller effectiveness exceeding 85 percent solve
    And Scoreboard 3 confirms unseen repository generalization with zero data leakage

  @domain
  Scenario: Cross-model frontier shift enables weaker models to beat stronger controls
    Given a multi-tier model evaluation across Weak, Medium, and Strong models
    When evaluated under Control versus MinTok
    Then a Weak model with MinTok achieves higher solve rate than a Medium model with Control
    And a Medium model with MinTok achieves higher solve rate than a Strong model with Control
    And the Inference Amplification metric exceeds 1.0 useful inference units per token

  @integration
  Scenario: CLI calibrate-sim runs validation on trajectory predictions
    When the user runs mintok calibrate-sim with json format
    Then the CLI exits with code 0
    And the report confirms simulator qualification status

  @integration
  Scenario: CLI adversarial-expanded runs 50-task benchmark
    When the user runs mintok adversarial-expanded with json format
    Then the CLI exits with code 0
    And the report includes 10 distinct failure mode families

  @integration
  Scenario: CLI ablation-ladder prints marginal decomposition
    When the user runs mintok ablation-ladder with json format
    Then the CLI exits with code 0
    And all 12 arms from A to L are reported

  @integration
  Scenario: CLI real-proof executes 100-task paired proof benchmark
    When the user runs mintok real-proof with json format
    Then the CLI exits with code 0
    And the report outputs the three scoreboards and cross-model shifts
