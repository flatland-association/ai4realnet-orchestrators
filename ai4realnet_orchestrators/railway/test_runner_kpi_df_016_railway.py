import logging
import os
from pathlib import Path

from flatland.trajectories.trajectories import Trajectory

from ai4realnet_orchestrators.railway.abstract_test_runner_railway import AbtractTestRunnerRailway


DATA_VOLUME_MOUNTPATH = os.environ.get("DATA_VOLUME_MOUNTPATH", "/app/data")
SCENARIOS_VOLUME_MOUNTPATH = os.environ.get("SCENARIOS_VOLUME_MOUNTPATH", "/app/scenarios")

DLA_DATA_URL = "ghcr.io/flatland-association/flatland-baselines-deadlock-avoidance-heuristic:latest"
DLA_POLICY_ARGS = [
    "--policy-pkg", "flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy",
    "--policy-cls", "DeadLockAvoidancePolicy",
    "--obs-builder-pkg", "flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation",
    "--obs-builder-cls", "FullEnvObservation",
]

logger = logging.getLogger(__name__)

# KPI-DF-016: Delay reduction efficiency (Railway)
class TestRunner_KPI_DF_016_Railway(AbtractTestRunnerRailway):

  def run_scenario(self, scenario_id: str, submission_id: str):
    env_path, seed = TestRunner_KPI_DF_016_Railway.load_scenario_data(scenario_id)

    run_args = [
      "--rewards-pkg", "flatland.envs.rewards", "--rewards-cls", "DelayRewards",
      "--malfunction-interval", "-1",
      "--post-seed", str(seed),
      "--effects-generator-pkg", "flatland.envs.malfunction_effects_generators", "--effects-generator-cls",
      "ConditionalMalfunctionEffectsGenerator",
      "--effects-generator-kwargs", "min_duration", "10",
      "--effects-generator-kwargs", "max_duration", "50",
      "--effects-generator-kwargs", "malfunction_rate", "0.01",
      "--effects-generator-kwargs", "condition_pkg", "flatland.envs.malfunction_effects_generators",
      "--effects-generator-kwargs", "condition_cls", "on_map_state_condition",
      "--ep-id", scenario_id,
      "--env-path", f"{SCENARIOS_VOLUME_MOUNTPATH}/{env_path}",
      "--snapshot-interval", "10",
    ]

    data_dir_dla = f"{DATA_VOLUME_MOUNTPATH}/{submission_id}/{self.test_id}/{scenario_id}/dla_baseline"
    orig_url = self.submission_data_url
    self.submission_data_url = DLA_DATA_URL
    self.exec(["--data-dir", data_dir_dla] + DLA_POLICY_ARGS + run_args, scenario_id, submission_id, f"{submission_id}/{self.test_id}/{scenario_id}/dla_baseline")
    self.submission_data_url = orig_url

    data_dir_submission = f"{DATA_VOLUME_MOUNTPATH}/{submission_id}/{self.test_id}/{scenario_id}/submission"
    self.exec(["--data-dir", data_dir_submission] + run_args, scenario_id, submission_id, f"{submission_id}/{self.test_id}/{scenario_id}/submission")

    # analyse dla baseline
    trajectory_dla = Trajectory.load_existing(data_dir=Path(data_dir_dla), ep_id=scenario_id)
    num_agents = trajectory_dla.trains_rewards_dones_infos["agent_id"].max() + 1
    tail_dla = trajectory_dla.trains_rewards_dones_infos.tail(num_agents)
    assert tail_dla["done"].values.sum() == num_agents
    delay_1 = tail_dla["reward"].sum()
    logger.info(f"delay dla baseline: {delay_1}")

    trajectory_submission = Trajectory.load_existing(data_dir=Path(data_dir_submission), ep_id=scenario_id)
    tail_submission = trajectory_submission.trains_rewards_dones_infos.tail(num_agents)
    assert tail_submission["done"].values.sum() == num_agents
    delay_2 = tail_submission["reward"].sum()
    logger.info(f"delay submission: {delay_2}")

    if delay_1 == 0:
        delay_reduction = 1.0 if delay_2 == 0 else 0.0
    else:
        delay_reduction = max(1 - delay_2 / delay_1, 0)
    
    self.upload_and_empty_local(submission_id=submission_id, scenario_id=scenario_id)
    
    return {
      'primary': delay_reduction,
      'punctuality_1': delay_1,
      'punctuality_2': delay_2,
    }  
    
  @staticmethod
  def load_scenario_data(scenario_id: str) -> list[str, int]:
    return {
      'bb6302f1-0dc2-43ed-976b-4e5d3126006a': ['Test_00/Level_0.pkl', 124],
      'ac3d32bf-2694-4405-953b-01849e7923ef': ['Test_00/Level_3.pkl', 490],
      '02e163b8-d8a3-44cb-9fb0-65501dfa35b7': ['Test_00/Level_6.pkl', 435],
      '7a3ae3eb-b783-44a3-80d4-aa9cb0bd55fb': ['Test_00/Level_9.pkl', 281],
      '01a82553-8d2c-4f84-94df-ccb9f3250734': ['Test_01/Level_2.pkl', 277],
      'db614cef-8b86-467d-a638-64c25a91ec78': ['Test_01/Level_5.pkl', 436],
      '3b68eeb2-96f6-4a87-8a2f-5decaf3cb3f0': ['Test_01/Level_8.pkl', 436],
      '7277b987-4cc2-4cb5-a308-bb226c832747': ['Test_02/Level_1.pkl', 121],
      '2b4b92d2-6871-4c20-ad58-11dc51718379': ['Test_02/Level_4.pkl', 498],
      'e5968696-5497-496b-8fe4-f40a837f7129': ['Test_02/Level_7.pkl', 211],
      '9b603e03-3e2e-4366-8127-96307d3b2ce1': ['Test_03/Level_0.pkl', 185],
      '51081d92-2ff1-40a4-b557-38215c125051': ['Test_03/Level_3.pkl', 151],
      '5832961b-942f-4d33-8614-c6dd4861ef46': ['Test_03/Level_6.pkl', 393],
      'e15fa3d0-0da3-4513-a5bd-6082806039a3': ['Test_03/Level_9.pkl', 313],
      '45b93b12-57cc-40ff-b277-82de8ceaec32': ['Test_04/Level_2.pkl', 474],
      '9ae1a2b5-fe89-4027-b1d8-8c3888862a5e': ['Test_04/Level_5.pkl', 421],
      '7b42a1cc-ce70-4d9a-804f-ac9027a1ee48': ['Test_04/Level_8.pkl', 234],
      'b30319f8-8953-4433-80b6-5b80c9103bc5': ['Test_05/Level_1.pkl', 241],
      '25f59eeb-3baf-4668-bdb1-2beb577fbf73': ['Test_05/Level_4.pkl', 429],
      'a35f5412-b565-4f24-9459-eb9ac1f7fe30': ['Test_05/Level_7.pkl', 224],
      'f262ffb3-86b9-4db8-8657-f4a96915cb83': ['Test_06/Level_0.pkl', 259],
      'fec35ca6-d093-4081-8cf1-2f3b8f445bed': ['Test_06/Level_3.pkl', 484],
      '14a20672-ea9c-4205-961a-4f2a3585eea2': ['Test_06/Level_6.pkl', 363],
      '63b933c7-b63c-4ab7-b602-69aa5f91aec9': ['Test_06/Level_9.pkl', 201],
      '4223ef16-0a01-404d-8024-5a656203d3f8': ['Test_07/Level_2.pkl', 213],
      '59b4935d-cc5d-4792-a395-770bae030d2d': ['Test_07/Level_5.pkl', 442],
      '10ee5c39-9ae9-4e58-bdc8-8a449887574d': ['Test_07/Level_8.pkl', 383],
      'dc4da56a-bf2d-4351-b280-d418736844de': ['Test_08/Level_1.pkl', 299],
      'd4b12805-017e-46d9-8fe7-220569a21477': ['Test_08/Level_4.pkl', 391],
      '6acbd77e-18a8-41c9-af81-1ff0ac1a9b0f': ['Test_08/Level_7.pkl', 310],
      'a7a37c14-b2a7-471c-9ed2-af90ee502d39': ['Test_09/Level_0.pkl', 145],
      '626f428b-0928-48c8-8770-12de6e3b18ed': ['Test_09/Level_3.pkl', 211],
      'd238a30f-2f16-4e12-83a6-fcf779cd7115': ['Test_09/Level_6.pkl', 349],
      'ac06a723-e801-470d-991d-b4411368ccfe': ['Test_09/Level_9.pkl', 185],
      '9eb9657e-5c48-461a-a680-7fae151f6800': ['Test_10/Level_2.pkl', 311],
      '1667a5c9-ae17-4628-ba55-74e34cb04332': ['Test_10/Level_5.pkl', 318],
      '2bb4384c-1119-4756-a2ad-fc2e5c00f952': ['Test_10/Level_8.pkl', 425],
      'c6ea3dff-9cda-471d-bf8b-a576b17036bb': ['Test_11/Level_1.pkl', 405],
      '03ffcc01-b691-406f-ac53-5d13e9328175': ['Test_11/Level_4.pkl', 436],
      '119b0c6b-4ce6-4a7f-b49b-ec904577a182': ['Test_11/Level_7.pkl', 438],
      'f3ee3bb9-3328-450a-98fb-63692042134f': ['Test_12/Level_0.pkl', 134],
      '6c90d7a4-980f-4e13-856c-13117e2edf82': ['Test_12/Level_3.pkl', 346],
      'f7509133-3083-4454-a963-95302eb66764': ['Test_12/Level_6.pkl', 160],
      'bf4447a8-9e3e-4b8c-b9ea-4bc5bb009977': ['Test_12/Level_9.pkl', 119],
      'd271e605-1b68-4609-884c-0e6b2417980a': ['Test_13/Level_2.pkl', 432],
      '7dd80b01-99c3-4a89-9b4b-f4c878a6d996': ['Test_13/Level_5.pkl', 104],
      'f9b8c0b3-0968-4324-9b92-35063c49def2': ['Test_13/Level_8.pkl', 342],
      '84db3ada-efe4-4cf7-bf05-3f14bbe2c668': ['Test_14/Level_1.pkl', 481],
      '05eeb2ea-67fe-405e-b630-43f382dbf246': ['Test_14/Level_4.pkl', 402],
      '90071fa0-a560-4c6e-b2ff-fd59588fbdb7': ['Test_14/Level_7.pkl', 295],
    }[scenario_id]