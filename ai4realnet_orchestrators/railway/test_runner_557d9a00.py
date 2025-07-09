import logging
import os

import pandas as pd

from ai4realnet_orchestrators.test_runner import TestRunner
from ai4realnet_orchestrators.fab_exec_utils import exec_with_logging

# required only for docker in docker
DATA_VOLUME = os.environ.get("DATA_VOLUME")
SUDO = os.environ.get("SUDO", "true").lower() == "true"

DATA_VOLUME_MOUNTPATH = os.environ.get("DATA_VOLUME_MOUNTPATH", "/app/data")


class TestRunner557d9a00(TestRunner):

  def run_scenario(self, scenario_id: str, submission_id: str):
    seed = TestRunner557d9a00.load_scenario_data(scenario_id)
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

    df_trains_arrived = pd.read_csv(f"{DATA_VOLUME_MOUNTPATH}/{submission_id}/{self.test_id}/{scenario_id}/event_logs/TrainMovementEvents.trains_arrived.tsv",
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
    return {'1ae61e4f-201b-4e97-a399-5c33fb75c57e': "42", '564ebb54-48f0-4837-8066-b10bb832af9d': "43"}[scenario_id]
