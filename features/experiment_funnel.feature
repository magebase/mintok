@domain
Feature: Experiment funnel
  A 120-task frontier run is the final exam, not the development loop. The
  funnel screens ideas before frontier compute is spent: static integrity
  checks first, a FAST-8 smoke suite next, then widening confirmation stages
  with sequential stopping, cached controls, and adaptive concurrency.

  The same verification discipline applies to every arm: a workspace with
  unverified edits never reaches the turn cap without a final verification
  attempt, in control and slicer alike.

  Scenario: Unverified edits are forced to verify at the turn cap
    Given a live loop with a 4-turn budget whose model only ever patches
    When the trajectory exhausts the budget without ever running the suite
    Then the harness runs the suite itself exactly once
    And the model gets one reaction turn after the forced verification
    And the forced suite call is recorded in the shim log

  Scenario: A verified trajectory is not forced to verify again
    Given a live loop with a 4-turn budget whose model runs the suite after each patch
    When the trajectory exhausts the budget
    Then the harness adds no forced verification

  Scenario: The FAST-8 suite covers the discriminating behavior classes
    Given the full task set contains the FAST-8 tasks
    When the FAST-8 suite is selected from the full task set
    Then it contains exactly 8 tasks
    And it covers a simple lookup task
    And it covers a cross-file bug task
    And it covers an api signature propagation task
    And it covers a schema or framework change task
    And it covers a large file navigation task
    And it covers a simple api extension task
    And it covers the semantic trap
    And it covers a feature addition task
    And every selected task exists in the full task set

  Scenario: A dominated arm is killed at the first checkpoint
    Given 5 paired tasks where the arm uses 1.5x the control tokens and matches solve rate
    When the sequential verdict is computed after the 5th pair
    Then the arm is killed
    And the reason mentions token ratio

  Scenario: A marginal arm survives early checkpoints but dies at 15
    Given 10 paired tasks where the arm uses 1.10x the control tokens and matches solve rate
    When the sequential verdict is computed after the 10th pair
    Then the arm continues
    When the arm reaches 15 paired tasks at 1.10x the control tokens
    When the sequential verdict is computed after the 15th pair
    Then the arm is killed
    And the reason mentions no meaningful improvement

  Scenario: A clearly better arm is never killed
    Given 15 paired tasks where the arm uses 0.70x the control tokens and matches solve rate
    When the sequential verdict is computed after the 15th pair
    Then the arm continues

  Scenario: A solve-rate regression kills the arm regardless of tokens
    Given 5 paired tasks where the arm uses 0.60x the control tokens and solves 20 points fewer
    When the sequential verdict is computed after the 5th pair
    Then the arm is killed
    And the reason mentions solve rate

  Scenario: Identical control configurations reuse the cached trajectory
    Given a completed control run for task "t1" with model "m" and effort "high"
    When a control run for the same task, model, effort, harness, and toolset is requested
    Then the cache serves the stored record without a new run
    When a control run for the same task with model "other" is requested
    Then the cache reports a miss

  Scenario: Immutable task fingerprints reject drift before any run
    Given a fingerprinted task with instruction, checker, and fixture files
    When the same task is verified against its fingerprint
    Then verification passes
    When the checker text is edited and the task is verified again
    Then verification fails with a drift error naming the changed part

  Scenario: The adaptive pool stays inside the provider budget
    When a fresh adaptive pool is created
    Then it allows 8 concurrent workers
    When three clean waves complete without throttling
    Then it allows 12 concurrent workers
    When the provider throttles a wave
    Then it allows 9 concurrent workers
    And it never exceeds 12 or drops below 4

  Scenario Outline: The funnel widens only on survivors
    Given an arm with verdict history "<history>"
    When the funnel advances the arm
    Then the arm is in phase "<phase>"

    Examples:
      | history            | phase       |
      | fresh              | smoke8      |
      | smoke8_pass        | dev15       |
      | dev15_pass         | confirm30   |
      | confirm30_pass     | eval120     |
      | smoke8_killed      | dead        |
      | dev15_killed       | dead        |
      | confirm30_killed   | dead        |

  Scenario: Slicer promotion verdict from measured trajectories
    Given slicer runs on 15 large-module tasks:
      | task | solved | tokens | turns | slice_tok | fallback_tok | expanded |
      | t1   | yes    | 700    | 4     | 300       | 200          | no       |
      | t2   | yes    | 900    | 5     | 250       | 400          | no       |
      | t3   | yes    | 600    | 3     | 350       | 100          | no       |
      | t4   | no     | 2100   | 12    | 400       | 1500         | yes      |
      | t5   | yes    | 500    | 4     | 300       | 100          | no       |
    When the slicer promotion verdict is computed with solve target 4
    Then the verdict is "promote"
    And the slice acceptance rate is 0.75
    And the raw-source fallback rate is 0.2
    And the expanded-slice rate is 0.2

  Scenario: A slicer that reads the file conventionally does not promote
    Given slicer runs on 15 large-module tasks:
      | task | solved | tokens | turns | slice_tok | fallback_tok | expanded |
      | t1   | yes    | 2400   | 8     | 300       | 1900         | no       |
      | t2   | yes    | 2600   | 9     | 300       | 2100         | no       |
      | t3   | yes    | 2200   | 7     | 300       | 1700         | no       |
      | t4   | yes    | 2500   | 8     | 300       | 2000         | no       |
    When the slicer promotion verdict is computed with solve target 4
    Then the verdict is "reject"

  Scenario: A frozen control may only be reused under an identical manifest
    Given a frozen control manifest with model "m1" and prompt hash "p1"
    When a live slicer manifest arrives with model "m1" and prompt hash "p1"
    Then the frozen control is compatible

    Given a frozen control manifest with model "m1" and prompt hash "p1"
    When a live slicer manifest arrives with model "m2" and prompt hash "p1"
    Then the frozen control is refused with reason "model"
    And the run demands fresh control trajectories

  Scenario: A missing frozen manifest demands fresh control
    Given no frozen control manifest exists
    When the comparison is planned
    Then the run demands fresh control trajectories

  Scenario: A checker validates its own task copy, never a cached one
    Given two task copies checked in sequence, both defining module "biglib"
    And the first copy's module state satisfies only the first checker
    When each copy's checker runs after its trajectory
    Then the second checker sees the second copy's module state
    And a checker that passes is recorded as solved regardless of position

  Scenario Outline: Failures are attributed by their evidence
    Given a failed slicer run where target_in_slice is <target>, slice_dominated is <dominated>, slice_truncated is <truncated>, edit_rejections is <rejects>, and suite_ok is <suite>
    Then the failure class is "<klass>"

    Examples:
      | target | dominated | truncated | rejects | suite | klass                 |
      | no     | no        | no        | 0       | no    | bad_slice             |
      | yes    | no        | yes       | 0       | no    | insufficient_slice    |
      | yes    | no        | no        | 0       | no    | fallback_needed       |
      | yes    | yes       | no        | 3       | no    | edit_tool_limitation  |
      | yes    | yes       | no        | 0       | yes   | checker_disagreement  |
      | yes    | yes       | no        | 0       | no    | agent_reasoning_failure |

  Scenario: The frozen holdout benchmark suite passes static fingerprint integrity
    When the holdout task suite is verified against its frozen fingerprints
    Then all 40 holdout tasks pass integrity verification

  Scenario: Benchmark workspace copies isolate solutions and checkers from the agent
    Given a holdout task prepared for benchmark execution
    When the workspace directory is inspected
    Then no solution files exist in the workspace
    And no checker source exists in the workspace

  Scenario: The frozen holdout arm schedule is strictly balanced 50/50
    When the frozen holdout arm schedule is loaded
    Then it specifies exactly 20 control-first tasks and 20 slicer-first tasks

  Scenario: Agent-exposed tools strictly reject paths escaping the workspace root
    Given a benchmark workspace with an external secret file and an escaping symlink
    When the agent attempts to read "../benchmarks/e2e/holdout_solutions.json"
    Then the tool call is rejected with an escaping root error
    When the agent attempts to read an absolute path to the external secret file
    Then the tool call is rejected with an escaping root error
    When the agent attempts to read the escaping symlink
    Then the tool call is rejected with an escaping root error
    When the agent attempts to patch "../benchmarks/e2e/holdout_solutions.json"
    Then the tool call is rejected with an escaping root error
    When the agent attempts to patch an absolute path to the external secret file
    Then the tool call is rejected with an escaping root error
    When the agent attempts to patch the escaping symlink
    Then the tool call is rejected with an escaping root error



