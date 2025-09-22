import logging
import os

import pandas as pd

from ai4realnet_orchestrators.fab_exec_utils import exec_with_logging
from ai4realnet_orchestrators.test_runner import TestRunner

# required only for docker in docker
DATA_VOLUME = os.environ.get("DATA_VOLUME")
SUDO = os.environ.get("SUDO", "true").lower() == "true"

DATA_VOLUME_MOUNTPATH = os.environ.get("DATA_VOLUME_MOUNTPATH", "/app/data")


# KPI-NF-045: Network Impact Propagation (Railway)
class TestRunner_KPI_NF_045_Railway(TestRunner):

    def run_scenario(self, scenario_id: str, submission_id: str):
        seed = TestRunner_KPI_NF_045_Railway.load_scenario_data(scenario_id)
        # here you would implement the logic to run the test for the scenario
        # data and other stuff initialized in the init method can be used here
        # for demonstration, we return a dummy result

        # --data-dir must exist -- TODO fix in flatland-rl instead
        args = ["docker", "run", "--rm", "-v", f"{DATA_VOLUME}:/vol", "alpine:latest", "mkdir", "-p", f"/vol/{submission_id}/{self.test_id}/{scenario_id}"]
        exec_with_logging(args if not SUDO else ["sudo"] + args)
        args = ["docker", "run", "--rm", "-v", f"{DATA_VOLUME}:/vol", "alpine:latest", "chmod", "-R", "a=rwx",
                f"/vol/{submission_id}/{self.test_id}/{scenario_id}"]
        exec_with_logging(args if not SUDO else ["sudo"] + args)
        args = [
            "docker", "run",
            "--rm",
            "-v", f"{DATA_VOLUME}:/app/data",
            "--entrypoint", "/bin/bash",
            # Don't allow subprocesses to raise privileges, see https://github.com/codalab/codabench/blob/43e01d4bc3de26e8339ddb1463eef7d960ddb3af/compute_worker/compute_worker.py#L520
            "--security-opt=no-new-privileges",
            # Don't buffer python output, so we don't lose any
            "-e", "PYTHONUNBUFFERED=1",
            # for integration tests with localhost http
            "-e", "OAUTHLIB_INSECURE_TRANSPORT=1",
            self.submission_data_url,
            # TODO hard-coded dependency on flatland-baselines
            "/home/conda/entrypoint_generic.sh", "flatland-trajectory-generate-from-policy",
            "--data-dir", f"/app/data/{submission_id}/{self.test_id}/{scenario_id}",
            "--policy-pkg", "flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy", "--policy-cls", "DeadLockAvoidancePolicy",
            "--obs-builder-pkg", "flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation", "--obs-builder-cls", "FullEnvObservation",
            "--ep-id", scenario_id,
            "--seed", seed,
        ]
        exec_with_logging(args if not SUDO else ["sudo"] + args, log_level_stdout=logging.DEBUG)

        df_trains_arrived = pd.read_csv(
            f"{DATA_VOLUME_MOUNTPATH}/{submission_id}/{self.test_id}/{scenario_id}/event_logs/TrainMovementEvents.trains_arrived.tsv",
            sep="\t")
        print(df_trains_arrived)
        assert len(df_trains_arrived) == 1
        print(df_trains_arrived.iloc[0])
        success_rate = df_trains_arrived.iloc[0]["success_rate"]
        print(success_rate)

        df_trains_rewards_dones_Infos = pd.read_csv(
            f"{DATA_VOLUME_MOUNTPATH}/{submission_id}/{self.test_id}/{scenario_id}/event_logs/TrainMovementEvents.trains_rewards_dones_infos.tsv",
            sep="\t")
        print(df_trains_rewards_dones_Infos)
        rewards = df_trains_rewards_dones_Infos["reward"].sum()
        print(rewards)

        return {
            'primary': rewards,
            'secondary': success_rate
        }

    @staticmethod
    def load_scenario_data(scenario_id: str) -> str:
        return {
          "bb6302f1-0dc2-43ed-976b-4e5d3126006a": "Test_00/Level_0.pkl",
          "270d3309-e184-499e-9423-130d1058a960": "Test_00/Level_1.pkl",
          "3496a668-c1ce-4136-8d4d-6c49375fe57b": "Test_00/Level_2.pkl",
          "f85e28b3-8cb0-4f13-92c1-1a66c55f5ecd": "Test_00/Level_3.pkl",
          "d296e6ae-fcbb-4b15-aef3-3383f2a247e8": "Test_00/Level_4.pkl",
          "b6802d87-f2ee-4831-a36c-25875f3f396e": "Test_00/Level_5.pkl",
          "0f1c945b-3ac3-4a64-887e-8ee085ca4946": "Test_00/Level_6.pkl",
          "311802c8-7054-43c1-abe2-0fd73bc56d0d": "Test_00/Level_7.pkl",
          "ad5f4515-4dc7-4ee1-b623-4cf906b53f92": "Test_00/Level_8.pkl",
          "4d3922de-dd9c-420d-a8a2-9813715fb09e": "Test_00/Level_9.pkl",
          "0ce45996-4904-4e61-aae9-43ba913801d1": "Test_01/Level_0.pkl",
          "de09edc0-64fa-4178-8e72-ca194f54ef6a": "Test_01/Level_1.pkl",
          "8809afec-07b8-4db1-b750-b7b8415e434d": "Test_01/Level_2.pkl",
          "64fe7f6f-0100-4de0-9ab8-13641050d955": "Test_01/Level_3.pkl",
          "e316c041-8c1e-4fdb-bcbf-a01261d8f70a": "Test_01/Level_4.pkl",
          "6481fb57-fdcc-4a4f-86c5-bb30cc0a321a": "Test_01/Level_5.pkl",
          "35faef61-3245-4a02-aebe-db14f34448eb": "Test_01/Level_6.pkl",
          "070de230-9781-44dc-a7f8-e2f3c815c037": "Test_01/Level_7.pkl",
          "713f3aab-2ff3-45e8-8684-ef5a55113b81": "Test_01/Level_8.pkl",
          "84f76fee-f6a3-4003-a947-68e86599ef88": "Test_01/Level_9.pkl",
          "f4492920-0b3f-4355-a54d-b4c3af12dc9f": "Test_02/Level_0.pkl",
          "90112580-ceb8-4fd8-9805-d82b026dae62": "Test_02/Level_1.pkl",
          "2af604ec-90e1-4479-95bc-ef761a044252": "Test_02/Level_2.pkl",
          "0bccba99-72d5-4553-95c4-4b37d0354026": "Test_02/Level_3.pkl",
          "85c09866-ac16-45c8-9302-a7d26a7e7429": "Test_02/Level_4.pkl",
          "4c8023d8-a13f-4c59-9b95-79c15d55a932": "Test_02/Level_5.pkl",
          "005f53af-2c2d-4c83-b8db-7749034ddabc": "Test_02/Level_6.pkl",
          "fefda326-d73a-4f5d-a974-bf7542b4570f": "Test_02/Level_7.pkl",
          "04fceab7-d26a-41d9-bc39-a3b31b455f86": "Test_02/Level_8.pkl",
          "0049f2fb-3bb8-40ae-8a31-5e94c04fa116": "Test_02/Level_9.pkl",
          "45c7d211-e1b8-499e-a1fc-05ebe213df25": "Test_03/Level_0.pkl",
          "49ff57b1-d3e3-4cfb-84e0-fb8a3e89ad4c": "Test_03/Level_1.pkl",
          "062fd640-1801-46c0-a031-43c19aaec1a8": "Test_03/Level_2.pkl",
          "c0692dd3-abb8-4b7a-9353-8fa120d2f61e": "Test_03/Level_3.pkl",
          "e694b225-c72c-4df7-9096-b24ef43f43d7": "Test_03/Level_4.pkl",
          "0ed40148-1ffb-4e3a-99b0-16bfd3b53fdf": "Test_03/Level_5.pkl",
          "06697084-7105-4d47-8aa6-dc364319e0ff": "Test_03/Level_6.pkl",
          "95ae7391-c54e-4a81-bae6-330d4b034147": "Test_03/Level_7.pkl",
          "2f70c0fb-de33-49db-8753-2c1eb46be5ad": "Test_03/Level_8.pkl",
          "2bfaab23-95d3-4213-9589-16dcde2356a2": "Test_03/Level_9.pkl",
          "f81ca2cf-d6a6-45f5-aac0-f3e28e1cb6dd": "Test_04/Level_0.pkl",
          "06793992-2217-4110-b82d-69eb360e5a77": "Test_04/Level_1.pkl",
          "793d606d-6965-4354-addf-6e3bedeb0b76": "Test_04/Level_2.pkl",
          "a5caac21-96d5-4f13-8fdf-38e0d9038021": "Test_04/Level_3.pkl",
          "8e3c5462-1e78-424f-8fc8-c001eaa820d6": "Test_04/Level_4.pkl",
          "4f27c075-6e67-4b84-a080-9bd9b8b7b26c": "Test_04/Level_5.pkl",
          "ed1b420a-9f46-4454-ba03-a1e371fee82c": "Test_04/Level_6.pkl",
          "fc6b585a-148e-4960-937a-aadddeb70b5b": "Test_04/Level_7.pkl",
          "0c8862ea-f611-4fe8-922f-8b518f8e4333": "Test_04/Level_8.pkl",
          "572c230e-3e60-42df-ab11-0e41b2d7f913": "Test_04/Level_9.pkl",
          "df8d18e7-3b9e-47b8-99b7-b93f8e736fd4": "Test_05/Level_0.pkl",
          "100dda89-4ca0-4f76-a8dd-2b9461ae0b95": "Test_05/Level_1.pkl",
          "02a3cc15-1a46-42e8-b535-6ded17ca3816": "Test_05/Level_2.pkl",
          "6a30d0a1-e19f-4003-9612-7a1aed5231e4": "Test_05/Level_3.pkl",
          "143064e0-a667-4133-9b1a-b26346b743ee": "Test_05/Level_4.pkl",
          "eb3142c2-53d8-41e8-8fad-d73da7c1325b": "Test_05/Level_5.pkl",
          "f3a7f733-d6d4-43eb-a92f-4442e910f955": "Test_05/Level_6.pkl",
          "90a57473-f35b-47c3-92ab-45f9eae5d3bf": "Test_05/Level_7.pkl",
          "4cad0c12-739b-45e8-a253-d7038c3d9e11": "Test_05/Level_8.pkl",
          "69e71442-b3c3-4654-a489-2e08dfbcbef5": "Test_05/Level_9.pkl",
          "4c09ca29-a43e-46f1-aaca-25299c3d6165": "Test_06/Level_0.pkl",
          "b5da59c9-6e8b-4fd6-8ca6-d7ac20a38968": "Test_06/Level_1.pkl",
          "a3a5a511-30d5-46f3-8e1d-03581eab426e": "Test_06/Level_2.pkl",
          "9d511fbd-f3fe-4a1b-b462-42c1fa3e260b": "Test_06/Level_3.pkl",
          "14e9d12f-6443-45b7-94bf-13316538746c": "Test_06/Level_4.pkl",
          "1b454ad9-d73b-42f7-86fc-fc1525ac982b": "Test_06/Level_5.pkl",
          "305e7004-0691-4f75-8e66-1ef7dff8ae34": "Test_06/Level_6.pkl",
          "7776b35a-8275-467c-86d6-6ae6a7b41364": "Test_06/Level_7.pkl",
          "05a069e6-5be2-4a2c-baa4-0b7f3a8402ae": "Test_06/Level_8.pkl",
          "a8e86c4c-1d1c-429e-af45-e576dbd7903b": "Test_06/Level_9.pkl",
          "3bcc8563-cd56-4de6-86d2-6e3a713c9246": "Test_07/Level_0.pkl",
          "59adb927-d7fd-4307-be34-c983afd68564": "Test_07/Level_1.pkl",
          "3c4c19e1-ba74-4e20-a913-6f2f48a86c76": "Test_07/Level_2.pkl",
          "09b6e8a6-895b-4226-804e-1566f55916c1": "Test_07/Level_3.pkl",
          "320290f4-bcfa-45e7-99f8-3027f921dcb9": "Test_07/Level_4.pkl",
          "fc1c9788-4dd2-44a1-95d7-74df0d75c0d3": "Test_07/Level_5.pkl",
          "ef6dfb89-0f1f-45e9-ad19-8c52fd698839": "Test_07/Level_6.pkl",
          "cca6ada1-f31b-4f11-b2bf-1eca5a155659": "Test_07/Level_7.pkl",
          "fcbbf106-3890-49e1-a090-deafadc943d9": "Test_07/Level_8.pkl",
          "05603ce7-139c-4948-bab2-506bce9dbcbd": "Test_07/Level_9.pkl",
          "b8677b45-84a1-482e-b7ab-3b9faf07cbd4": "Test_08/Level_0.pkl",
          "52977b37-121c-486b-aa1f-310158a6ca27": "Test_08/Level_1.pkl",
          "728379a4-6439-4e30-9085-b4f5d9582045": "Test_08/Level_2.pkl",
          "139c060e-8949-475f-badf-2f964a0dd045": "Test_08/Level_3.pkl",
          "5e9626d5-2f61-4e06-8e9e-684fa0e8fc03": "Test_08/Level_4.pkl",
          "97a8d51a-3f23-4137-97b5-502b33f28d50": "Test_08/Level_5.pkl",
          "8c26dc67-c465-4c55-9182-fe35c19013a0": "Test_08/Level_6.pkl",
          "36a72d16-5a73-48f2-a23a-8ae12ca2abd4": "Test_08/Level_7.pkl",
          "60e8bed4-2ed7-40e9-9e17-fa4010d6cf74": "Test_08/Level_8.pkl",
          "d47291ba-670e-47cb-8ea0-e1e785555c58": "Test_08/Level_9.pkl",
          "4826b248-c86f-45dd-b08e-70a9039f4e84": "Test_09/Level_0.pkl",
          "54bc44ad-b519-4e77-b677-6135634644d4": "Test_09/Level_1.pkl",
          "18656965-cdb7-48d9-8d5b-7db0b166e4a9": "Test_09/Level_2.pkl",
          "6515ca6a-d53c-4ba4-a1cb-c5959a251716": "Test_09/Level_3.pkl",
          "c580edff-4e44-4a2f-adb3-0f9323fdcb40": "Test_09/Level_4.pkl",
          "5ec109a5-b176-43a6-a4a0-a466b42554ea": "Test_09/Level_5.pkl",
          "43279f74-a2a1-401a-9b60-6e86cc4bf2f6": "Test_09/Level_6.pkl",
          "9b373159-165e-425a-9a81-cf7554b4c008": "Test_09/Level_7.pkl",
          "7499ffb1-2412-4cad-90f3-9f4ace43276f": "Test_09/Level_8.pkl",
          "ba2d4a2d-c82a-4fa8-ae60-1b4db2fcef0f": "Test_09/Level_9.pkl",
          "6f2e3af3-439b-4dda-8c9f-3a597a8b513b": "Test_10/Level_0.pkl",
          "afbba707-ec0e-4236-94ad-42139c524335": "Test_10/Level_1.pkl",
          "22350a8a-304b-479a-b5a9-cf5e4d72aeb5": "Test_10/Level_2.pkl",
          "3d141a06-c5f0-4863-b305-09cd29b459a9": "Test_10/Level_3.pkl",
          "cdd4c660-5066-4792-92f0-beecb6ecc4b3": "Test_10/Level_4.pkl",
          "4fee92f4-d4b2-43c6-b560-6b56641df6b6": "Test_10/Level_5.pkl",
          "cdcf4336-4c20-4b3f-ae8b-0d9c9e617617": "Test_10/Level_6.pkl",
          "ad0e60f8-20cc-4120-b929-dcb80d10490a": "Test_10/Level_7.pkl",
          "240d80de-9906-48c5-94be-5dbc56caddc3": "Test_10/Level_8.pkl",
          "683c600e-38aa-4d33-91d9-3603c2f3b415": "Test_10/Level_9.pkl",
          "66f501d6-3e85-434f-8530-dce67483e70f": "Test_11/Level_0.pkl",
          "97a0e819-0a0d-4b03-8016-b54fdc2765f5": "Test_11/Level_1.pkl",
          "9947d18b-b31f-4347-8b67-185e4dc3b798": "Test_11/Level_2.pkl",
          "5e43286e-d994-413b-be14-cb7bb6efc94a": "Test_11/Level_3.pkl",
          "9b83c7b6-40dc-4215-a230-d96198cb47af": "Test_11/Level_4.pkl",
          "82b83b6c-c4ee-4992-bae5-c18f198e8f87": "Test_11/Level_5.pkl",
          "8bfe4190-bda1-4c1f-9d4c-6fe6e658ecbb": "Test_11/Level_6.pkl",
          "3b2a8afe-e853-4223-ae8e-eafaf366c2e8": "Test_11/Level_7.pkl",
          "49cf139a-096b-4985-bc27-fc029f353013": "Test_11/Level_8.pkl",
          "e0cd9286-2d3e-474e-b435-d7cfa0014e0a": "Test_11/Level_9.pkl",
          "be3c2375-4152-4408-a669-4df83ce96223": "Test_12/Level_0.pkl",
          "cf982b04-a900-4df4-aa74-7a7fc2fa6a35": "Test_12/Level_1.pkl",
          "58836148-ed88-4fb4-a561-75432759491d": "Test_12/Level_2.pkl",
          "5b2885b6-820d-4f70-9253-ece24d9839d7": "Test_12/Level_3.pkl",
          "6eeb2d19-df06-4946-a48c-cf5c2ed14aa8": "Test_12/Level_4.pkl",
          "0b969620-e9cb-4a4e-9767-f8a4698ebee4": "Test_12/Level_5.pkl",
          "37110ff9-9c0f-4e2e-97c5-f615bef73c06": "Test_12/Level_6.pkl",
          "60b24050-0c6a-408d-9acb-63a2c46b2b27": "Test_12/Level_7.pkl",
          "bb9532af-16be-411c-b927-4ba3ff53f7e6": "Test_12/Level_8.pkl",
          "df9a03ef-5a11-42a2-921f-e3615e0529e0": "Test_12/Level_9.pkl",
          "3d557f25-a38e-4104-a0d3-a24df04ed95b": "Test_13/Level_0.pkl",
          "b7abb233-2471-4c86-a3cf-0c8344887784": "Test_13/Level_1.pkl",
          "83ebef0f-85d2-426c-8c17-4c934805dc40": "Test_13/Level_2.pkl",
          "b883cbdd-8413-4ff6-9ac0-12e2856bbb06": "Test_13/Level_3.pkl",
          "f08b4d45-ec78-4f33-9070-f4046a0699fc": "Test_13/Level_4.pkl",
          "300a415d-f0d8-4fe8-985f-d35591db836b": "Test_13/Level_5.pkl",
          "a1c8b889-19ad-40a1-99a3-1379dbb997fe": "Test_13/Level_6.pkl",
          "443073bd-1d38-4221-bad5-40f6286ed302": "Test_13/Level_7.pkl",
          "177c68c3-4ea4-4b27-b8a3-8175b70177e8": "Test_13/Level_8.pkl",
          "71040e3b-c8f3-4583-8010-b2e17babc04a": "Test_13/Level_9.pkl",
          "243577d2-8f49-41c3-826e-78f1978308df": "Test_14/Level_0.pkl",
          "910798c7-e257-45d7-b6c5-ac780b8b6b94": "Test_14/Level_1.pkl",
          "4f2bf7e1-5cd9-4e58-9e62-85e05c4b8329": "Test_14/Level_2.pkl",
          "add4ac48-5c7e-4e59-99e5-ea73020a08f0": "Test_14/Level_3.pkl",
          "b4851a29-56ca-4403-8976-c01e03a9cf09": "Test_14/Level_4.pkl",
          "f79f21b5-3941-4f29-bd68-55dfcc32c8fd": "Test_14/Level_5.pkl",
          "f13e1240-1062-4273-9478-baab7cc462b1": "Test_14/Level_6.pkl",
          "f16ee6bb-ed6d-480e-8956-ed825d3b65a9": "Test_14/Level_7.pkl",
          "7e1aeaf4-bcbf-4fe1-a5d6-f4e13cc54275": "Test_14/Level_8.pkl",
          "1e46c049-9072-4019-9333-75145ba67e2b": "Test_14/Level_9.pkl"
        }[scenario_id]
