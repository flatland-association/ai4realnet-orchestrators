import ast
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from flatland.envs.step_utils.states import TrainState
from flatland.trajectories.trajectories import Trajectory

from ai4realnet_orchestrators.railway.abstract_test_runner_railway import AbtractTestRunnerRailway

DATA_VOLUME_MOUNTPATH = os.environ.get("DATA_VOLUME_MOUNTPATH", "/app/data")
SCENARIOS_VOLUME_MOUNTPATH = os.environ.get("SCENARIOS_VOLUME_MOUNTPATH", "/app/scenarios")

logger = logging.getLogger(__name__)


# KPI-DF-016: Delay reduction efficiency (Railway)
class TestRunner_KPI_DF_016_Railway(AbtractTestRunnerRailway):

  def run_scenario(self, scenario_id: str, submission_id: str):
    env_path = TestRunner_KPI_DF_016_Railway.load_scenario_data(scenario_id)

    data_dir_dla = f"{DATA_VOLUME_MOUNTPATH}/{submission_id}/{self.test_id}/{scenario_id}/with_malfunction"  # change to dla -> could this be run once and saved s.t. it does not have to be executed für every submission?
    generate_policy_args_dla_baseline = [
      "--data-dir", data_dir_dla,
      "--rewards-pkg", "flatland.envs.rewards", "--rewards-cls", "DelayRewards",
      "--malfunction-interval", "-1",
      "--effects-generator-pkg", "flatland.envs.malfunction_effects_generators", "--effects-generator-cls",
      "ConditionalMalfunctionEffectsGenerator",
      "--effects-generator-kwargs", "min_duration", "20",
      "--effects-generator-kwargs", "max_duration", "50",
      "--effects-generator-kwargs", "malfunction_rate", "0.1",  # what is a meaningful rate here?
      "--effects-generator-kwargs", "condition_pkg", "flatland.envs.malfunction_effects_generators",
      "--effects-generator-kwargs", "condition_cls", "on_map_state_condition",
      "--ep-id", scenario_id,
      "--env-path", f"{SCENARIOS_VOLUME_MOUNTPATH}/{env_path}",
      "--snapshot-interval", "10",
    ]
    self.exec(generate_policy_args_dla_baseline, scenario_id, submission_id, f"{submission_id}/{self.test_id}/{scenario_id}/with_malfunction")

    data_dir_submission = f"{DATA_VOLUME_MOUNTPATH}/{submission_id}/{self.test_id}/{scenario_id}/with_malfunction"
    generate_policy_args_submission = [
      "--data-dir", data_dir_submission,
      "--rewards-pkg", "flatland.envs.rewards", "--rewards-cls", "DelayRewards",
      "--malfunction-interval", "-1",
      "--effects-generator-pkg", "flatland.envs.malfunction_effects_generators", "--effects-generator-cls",
      "ConditionalMalfunctionEffectsGenerator",
      "--effects-generator-kwargs", "min_duration", "20",
      "--effects-generator-kwargs", "max_duration", "50",
      "--effects-generator-kwargs", "malfunction_rate", "0.1",  # what is a meaningful rate here?
      "--effects-generator-kwargs", "condition_pkg", "flatland.envs.malfunction_effects_generators",
      "--effects-generator-kwargs", "condition_cls", "on_map_state_condition",
      "--ep-id", scenario_id,
      "--env-path", f"{SCENARIOS_VOLUME_MOUNTPATH}/{env_path}",
      "--snapshot-interval", "10",
    ]
    self.exec(generate_policy_args_submission, scenario_id, submission_id, f"{submission_id}/{self.test_id}/{scenario_id}/with_malfunction")

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

    delay_reduction = max(1 - delay_2 / delay_1, 0)

    self.upload_and_empty_local(submission_id=submission_id, scenario_id=scenario_id)

    return {
      'primary': delay_reduction,
      'punctuality_1': delay_1,
      'punctuality_2': delay_2,
    }

  @staticmethod
  def load_scenario_data(scenario_id: str) -> str:
    return {
      "bb6302f1-0dc2-43ed-976b-4e5d3126006a": ["Test_00/Level_0.pkl", 20],
      "f84dcf0c-4bde-460b-9139-ea76e3694267": ["Test_00/Level_1.pkl", 22],
      "89ea38d1-e42e-430e-8a72-f426f1cc0be7": ["Test_00/Level_2.pkl", 20],
      "ac3d32bf-2694-4405-953b-01849e7923ef": ["Test_00/Level_3.pkl", 20],
      "30286226-29a3-4aa6-8243-562b88967d76": ["Test_00/Level_4.pkl", 38],
      "18276866-5a94-412b-b09c-9cac2ca5add0": ["Test_00/Level_5.pkl", 13],
      "02e163b8-d8a3-44cb-9fb0-65501dfa35b7": ["Test_00/Level_6.pkl", 22],
      "ab2b11c8-66f4-47c3-9cd3-f765eb772dc7": ["Test_00/Level_7.pkl", 13],
      "f3ae4180-86f3-409a-a51e-c1deb7e005cd": ["Test_00/Level_8.pkl", 24],
      "7a3ae3eb-b783-44a3-80d4-aa9cb0bd55fb": ["Test_00/Level_9.pkl", 25],
      "cff75f1a-8ea2-4f1d-b516-60dd0d625fe1": ["Test_01/Level_0.pkl", 20],
      "aa4fd74f-4680-405b-a184-c9392f9218e3": ["Test_01/Level_1.pkl", 22],
      "01a82553-8d2c-4f84-94df-ccb9f3250734": ["Test_01/Level_2.pkl", 20],
      "70316412-5480-44ca-9c2b-c51426b0390e": ["Test_01/Level_3.pkl", 20],
      "60a6acda-9a1a-4a0a-8c04-75de02304713": ["Test_01/Level_4.pkl", 38],
      "db614cef-8b86-467d-a638-64c25a91ec78": ["Test_01/Level_5.pkl", 13],
      "43b053bb-5e9b-4538-a490-fee839344203": ["Test_01/Level_6.pkl", 22],
      "e01032e4-2047-455a-a329-175a40a8de24": ["Test_01/Level_7.pkl", 13],
      "3b68eeb2-96f6-4a87-8a2f-5decaf3cb3f0": ["Test_01/Level_8.pkl", 24],
      "fdd89c15-3f8d-4381-9cd7-e8b773d06997": ["Test_01/Level_9.pkl", 25],
      "1d8f2bda-38a4-41de-a614-291b9e4697e4": ["Test_02/Level_0.pkl", 33],
      "7277b987-4cc2-4cb5-a308-bb226c832747": ["Test_02/Level_1.pkl", 36],
      "62e20486-eb7f-49d9-a9dc-7aa00fdfefb0": ["Test_02/Level_2.pkl", 47],
      "ae7a8233-8a80-496a-a2b3-0afd9a28ebe6": ["Test_02/Level_3.pkl", 26],
      "2b4b92d2-6871-4c20-ad58-11dc51718379": ["Test_02/Level_4.pkl", 54],
      "86f360de-8c4d-44d0-b089-3259a91dc3ea": ["Test_02/Level_5.pkl", 35],
      "f4b1aaeb-a498-428e-8f8f-2ed07aee0641": ["Test_02/Level_6.pkl", 52],
      "e5968696-5497-496b-8fe4-f40a837f7129": ["Test_02/Level_7.pkl", 30],
      "dec5cd4e-10b7-4a7e-a803-10e50badaaf5": ["Test_02/Level_8.pkl", 33],
      "139b31bd-22e4-495c-8e21-5e6e34cd5a20": ["Test_02/Level_9.pkl", 40],
      "9b603e03-3e2e-4366-8127-96307d3b2ce1": ["Test_03/Level_0.pkl", 46],
      "54601145-edd9-469f-8180-245e26dff069": ["Test_03/Level_1.pkl", 47],
      "34fa69f7-e0f7-4fd4-adf0-2ed8a47d6abc": ["Test_03/Level_2.pkl", 26],
      "51081d92-2ff1-40a4-b557-38215c125051": ["Test_03/Level_3.pkl", 29],
      "fef8ec79-80da-4039-9484-6ec49a29263e": ["Test_03/Level_4.pkl", 23],
      "d9d80121-bf6c-41ee-bc18-dade0e853ada": ["Test_03/Level_5.pkl", 46],
      "5832961b-942f-4d33-8614-c6dd4861ef46": ["Test_03/Level_6.pkl", 52],
      "e44e2b83-ad54-4e9c-a2ea-c23c1a249c54": ["Test_03/Level_7.pkl", 27],
      "8ca33f45-3839-4ccc-aa88-146b41fee9f3": ["Test_03/Level_8.pkl", 38],
      "e15fa3d0-0da3-4513-a5bd-6082806039a3": ["Test_03/Level_9.pkl", 43],
      "e28dc7e5-03ae-4687-ba37-c7ed5914c901": ["Test_04/Level_0.pkl", 55],
      "ef52b0f5-a147-4333-9817-fbd7e53143ee": ["Test_04/Level_1.pkl", 45],
      "45b93b12-57cc-40ff-b277-82de8ceaec32": ["Test_04/Level_2.pkl", 49],
      "ae557fe8-2155-42b3-8d17-2e9de47dda4b": ["Test_04/Level_3.pkl", 44],
      "3b2f7caf-2e32-4db9-8377-e01f50e436c2": ["Test_04/Level_4.pkl", 55],
      "9ae1a2b5-fe89-4027-b1d8-8c3888862a5e": ["Test_04/Level_5.pkl", 41],
      "9c221d41-fda7-409e-9ceb-a0f94018a92c": ["Test_04/Level_6.pkl", 48],
      "7da98e75-8c84-4cfc-98f4-0fedf1aec08f": ["Test_04/Level_7.pkl", 44],
      "7b42a1cc-ce70-4d9a-804f-ac9027a1ee48": ["Test_04/Level_8.pkl", 41],
      "48caf228-64f6-4b03-ad20-5a34cf8dd2ee": ["Test_04/Level_9.pkl", 50],
      "49cac9a9-1aac-4542-a01d-6483052bf02b": ["Test_05/Level_0.pkl", 67],
      "b30319f8-8953-4433-80b6-5b80c9103bc5": ["Test_05/Level_1.pkl", 67],
      "401a5b54-feb8-4eaf-92c3-426cb2f221ef": ["Test_05/Level_2.pkl", 71],
      "06863bb7-48d5-4897-87c7-3328546efdef": ["Test_05/Level_3.pkl", 66],
      "25f59eeb-3baf-4668-bdb1-2beb577fbf73": ["Test_05/Level_4.pkl", 50],
      "2a631e96-a912-4b27-b82e-57ca3dd4aacf": ["Test_05/Level_5.pkl", 55],
      "a85cd328-09f9-4360-ae04-4479301b5987": ["Test_05/Level_6.pkl", 73],
      "a35f5412-b565-4f24-9459-eb9ac1f7fe30": ["Test_05/Level_7.pkl", 74],
      "2060f4fe-4f43-4095-b14f-a3c8ce312a42": ["Test_05/Level_8.pkl", 73],
      "4e2e9ee7-26e1-4a2c-bc3c-93761a0ea43c": ["Test_05/Level_9.pkl", 82],
      "f262ffb3-86b9-4db8-8657-f4a96915cb83": ["Test_06/Level_0.pkl", 81],
      "b022b575-103d-4ded-8da5-2e9a8f686da6": ["Test_06/Level_1.pkl", 79],
      "ba5308f8-7e12-4c6e-8e4f-42f6280f537c": ["Test_06/Level_2.pkl", 95],
      "fec35ca6-d093-4081-8cf1-2f3b8f445bed": ["Test_06/Level_3.pkl", 82],
      "a410e586-219d-402f-9e34-6a1720ae46bc": ["Test_06/Level_4.pkl", 77],
      "a82beae8-9b1a-4a00-bea2-4891b56f3014": ["Test_06/Level_5.pkl", 101],
      "14a20672-ea9c-4205-961a-4f2a3585eea2": ["Test_06/Level_6.pkl", 70],
      "e7f82820-1caf-4b4f-ae70-8ea4d95dcb0c": ["Test_06/Level_7.pkl", 112],
      "989cc7e2-1d39-4334-8130-b704fd7c6c7b": ["Test_06/Level_8.pkl", 125],
      "63b933c7-b63c-4ab7-b602-69aa5f91aec9": ["Test_06/Level_9.pkl", 58],
      "05826982-694c-4ba3-817f-979d69942d36": ["Test_07/Level_0.pkl", 64],
      "dc280d8b-d3ca-4517-9d31-9e70e2f3aea8": ["Test_07/Level_1.pkl", 96],
      "4223ef16-0a01-404d-8024-5a656203d3f8": ["Test_07/Level_2.pkl", 87],
      "320b95b2-84d6-4827-ba37-0de57a1e6360": ["Test_07/Level_3.pkl", 78],
      "b9164bdc-9f5e-42b8-8379-7b6f454a3933": ["Test_07/Level_4.pkl", 92],
      "59b4935d-cc5d-4792-a395-770bae030d2d": ["Test_07/Level_5.pkl", 88],
      "be970bda-465f-42b4-9223-c4ba742b24ea": ["Test_07/Level_6.pkl", 89],
      "8df90815-5b7b-46c6-b388-546efbfa18a0": ["Test_07/Level_7.pkl", 110],
      "10ee5c39-9ae9-4e58-bdc8-8a449887574d": ["Test_07/Level_8.pkl", 120],
      "23ce8d72-6c32-45b4-a04e-34e029eb509e": ["Test_07/Level_9.pkl", 111],
      "f185083b-3f74-4221-b5c8-7b2e561ae2e4": ["Test_08/Level_0.pkl", 108],
      "dc4da56a-bf2d-4351-b280-d418736844de": ["Test_08/Level_1.pkl", 104],
      "02183d8b-2328-4467-81e9-97afde5618c9": ["Test_08/Level_2.pkl", 131],
      "d85cbc22-951e-4926-94f9-9c7a703b54eb": ["Test_08/Level_3.pkl", 106],
      "d4b12805-017e-46d9-8fe7-220569a21477": ["Test_08/Level_4.pkl", 125],
      "2f5a8e75-521e-4398-bfd2-e7ff7c9e0be4": ["Test_08/Level_5.pkl", 96],
      "701c5b31-06d0-4e54-82b9-08e1612f1042": ["Test_08/Level_6.pkl", 99],
      "6acbd77e-18a8-41c9-af81-1ff0ac1a9b0f": ["Test_08/Level_7.pkl", 195],
      "20ef1912-26a4-48c1-ad1b-f08c95b144e2": ["Test_08/Level_8.pkl", 90],
      "890c8007-a763-4a94-86d5-28d8c11c573f": ["Test_08/Level_9.pkl", 121],
      "a7a37c14-b2a7-471c-9ed2-af90ee502d39": ["Test_09/Level_0.pkl", 166],
      "848a39f3-e96b-4c41-83b8-78a2eb99403f": ["Test_09/Level_1.pkl", 159],
      "5dbc518b-6a94-4e7f-b140-f99eb30af9b2": ["Test_09/Level_2.pkl", 194],
      "626f428b-0928-48c8-8770-12de6e3b18ed": ["Test_09/Level_3.pkl", 137],
      "6129f9a8-853c-459e-8ac2-aa7cbc65802e": ["Test_09/Level_4.pkl", 173],
      "5869ddd8-fbb9-431e-beb7-64761220e3e3": ["Test_09/Level_5.pkl", 211],
      "d238a30f-2f16-4e12-83a6-fcf779cd7115": ["Test_09/Level_6.pkl", 301],
      "e86883e7-53be-4b58-99cb-efd5f23cdb5b": ["Test_09/Level_7.pkl", 124],
      "08c0e125-3ee8-45d4-94fc-84aa5d8c711d": ["Test_09/Level_8.pkl", 169],
      "ac06a723-e801-470d-991d-b4411368ccfe": ["Test_09/Level_9.pkl", 246],
      "566f099a-2957-4fc9-8e13-f9564311ba33": ["Test_10/Level_0.pkl", 146],
      "b990e1ab-5a11-4eba-9719-3b8b77846365": ["Test_10/Level_1.pkl", 145],
      "9eb9657e-5c48-461a-a680-7fae151f6800": ["Test_10/Level_2.pkl", 158],
      "2ff0136f-7995-4134-8e4e-9fd92da46ea8": ["Test_10/Level_3.pkl", 174],
      "fb08a40e-b5cc-4f8a-9b72-c9fe901fcb04": ["Test_10/Level_4.pkl", 187],
      "1667a5c9-ae17-4628-ba55-74e34cb04332": ["Test_10/Level_5.pkl", 162],
      "387dafe7-97ad-4b37-88e4-10785748e04d": ["Test_10/Level_6.pkl", 154],
      "6735f34e-d3a1-41d5-86a0-f623099bf2cf": ["Test_10/Level_7.pkl", 214],
      "2bb4384c-1119-4756-a2ad-fc2e5c00f952": ["Test_10/Level_8.pkl", 159],
      "5a52729c-4f09-4f58-852d-48239e9ce217": ["Test_10/Level_9.pkl", 149],
      "4c6e1a87-a083-46c1-b928-0c1d1763a9db": ["Test_11/Level_0.pkl", 161],
      "c6ea3dff-9cda-471d-bf8b-a576b17036bb": ["Test_11/Level_1.pkl", 177],
      "49257861-320d-4aa2-aa62-b9c4e1751791": ["Test_11/Level_2.pkl", 291],
      "be2823fd-5389-4415-9447-ed84006cee4c": ["Test_11/Level_3.pkl", 156],
      "03ffcc01-b691-406f-ac53-5d13e9328175": ["Test_11/Level_4.pkl", 244],
      "220bf75b-d271-4c46-bde2-67b313f10d13": ["Test_11/Level_5.pkl", 177],
      "eb2a0321-e6d4-49c7-8885-7e046018e12d": ["Test_11/Level_6.pkl", 207],
      "119b0c6b-4ce6-4a7f-b49b-ec904577a182": ["Test_11/Level_7.pkl", 168],
      "492c01f3-a7d4-46e5-b140-e82a22b13cdf": ["Test_11/Level_8.pkl", 175],
      "45107995-4a41-4831-98c9-df7ed79734b1": ["Test_11/Level_9.pkl", 131],
      "f3ee3bb9-3328-450a-98fb-63692042134f": ["Test_12/Level_0.pkl", 263],
      "d9f53db5-8cec-4ec4-8d62-d116cb198811": ["Test_12/Level_1.pkl", 236],
      "0c2f1499-dbbe-43df-be32-f3e980ac1691": ["Test_12/Level_2.pkl", 225],
      "6c90d7a4-980f-4e13-856c-13117e2edf82": ["Test_12/Level_3.pkl", 283],
      "7a335c56-af9e-4562-a614-bfe84a75951f": ["Test_12/Level_4.pkl", 304],
      "6267d830-501a-4646-823e-adab3403faf0": ["Test_12/Level_5.pkl", 287],
      "f7509133-3083-4454-a963-95302eb66764": ["Test_12/Level_6.pkl", 284],
      "bb2b5fe3-92ed-4270-9235-dafbfffa2d03": ["Test_12/Level_7.pkl", 253],
      "9b3fc207-3e88-4ef8-90b0-8ab1e77ea932": ["Test_12/Level_8.pkl", 216],
      "bf4447a8-9e3e-4b8c-b9ea-4bc5bb009977": ["Test_12/Level_9.pkl", 293],
      "42b7c577-c4ba-43e0-8ff9-d1839d86a06c": ["Test_13/Level_0.pkl", 275],
      "ae32ef91-586e-4dfd-b42b-e76002335794": ["Test_13/Level_1.pkl", 214],
      "d271e605-1b68-4609-884c-0e6b2417980a": ["Test_13/Level_2.pkl", 229],
      "ef64aa0d-ba0d-4af4-abf3-22395713f0d7": ["Test_13/Level_3.pkl", 438],
      "2f289e62-3b78-4580-90ed-c120947f70b5": ["Test_13/Level_4.pkl", 274],
      "7dd80b01-99c3-4a89-9b4b-f4c878a6d996": ["Test_13/Level_5.pkl", 229],
      "56e0d8cc-b7f0-4e9f-ae77-a7322cc9a2a5": ["Test_13/Level_6.pkl", 341],
      "ac523ef0-6a7a-470d-a944-795305899de9": ["Test_13/Level_7.pkl", 261],
      "f9b8c0b3-0968-4324-9b92-35063c49def2": ["Test_13/Level_8.pkl", 315],
      "8540245e-c841-4caa-aeb7-5dd6955bc43d": ["Test_13/Level_9.pkl", 268],
      "b7a9d4ba-c51f-452d-9128-3f923906ec18": ["Test_14/Level_0.pkl", 261],
      "84db3ada-efe4-4cf7-bf05-3f14bbe2c668": ["Test_14/Level_1.pkl", 290],
      "c7864f2d-ae67-42a9-bddb-59b6933b0c1c": ["Test_14/Level_2.pkl", 264],
      "4a9d8df0-d87b-4e31-9762-739f03694828": ["Test_14/Level_3.pkl", 271],
      "05eeb2ea-67fe-405e-b630-43f382dbf246": ["Test_14/Level_4.pkl", 378],
      "87e0d07c-cd5f-4506-b112-619d298ce924": ["Test_14/Level_5.pkl", 293],
      "c3b92403-d342-4d2e-b107-ae4ab443798f": ["Test_14/Level_6.pkl", 277],
      "90071fa0-a560-4c6e-b2ff-fd59588fbdb7": ["Test_14/Level_7.pkl", 272],
      "97bbc19c-de0c-4deb-838d-5675d9525eb8": ["Test_14/Level_8.pkl", 258],
      "cb55a7a4-460a-48e6-a623-4ebbc88b7be7": ["Test_14/Level_9.pkl", 356],
    }[scenario_id]