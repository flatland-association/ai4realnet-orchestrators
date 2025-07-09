import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd



from fab_exec_utils import exec_with_logging
def run_and_evaluate_test_557d9a00(submission_id: str, test_id: str, submission_data_url: str) -> Any:
  """
  Run experiment and upload results.

  Parameters
  ----------
  submission_id: specifies the submission
  test_id: specifies the test to execute
  submission_data_url: reference to the prediction module to use
  """

  # required only for docker in docker
  DATA_VOLUME = os.environ.get("DATA_VOLUME")

  # sudo required when doing DinD - otherwise, we get "permission denied while trying to connect to the Docker daemon socket"
  SUDO = os.environ.get("SUDO", "true").lower() == "true"
  DATA_VOLUME_MOUNTPATH = os.environ.get("DATA_VOLUME_MOUNTPATH", "/app/data")

  # --data-dir must exist -- TODO fix in flatland-rl instead
  args = ["docker", "run", "--rm", "-v", f"{DATA_VOLUME}:/vol", "alpine:latest", "mkdir", "-p", f"/vol/{test_id}/{submission_id}"]
  exec_with_logging(args if not SUDO else ["sudo"] + args)
  args = ["docker", "run", "--rm", "-v", f"{DATA_VOLUME}:/vol", "alpine:latest", "chmod", "-R", "a=rwx",
          f"/vol/{test_id}/{submission_id}"]
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
    submission_data_url,
    # TODO hard-coded dependency on flatland-baselines
    "/home/conda/entrypoint_generic.sh", "flatland-trajectory-generate-from-policy",
    "--data-dir", f"/app/data/{test_id}/{submission_id}",
    "--policy-pkg", "flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy", "--policy-cls", "DeadLockAvoidancePolicy",
    "--obs-builder-pkg", "flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation", "--obs-builder-cls", "FullEnvObservation",
    "--ep-id", submission_id
  ]
  exec_with_logging(args if not SUDO else ["sudo"] + args, log_level_stdout=logging.DEBUG)



  # run your experiment here and write results to "@TestId.json"
  df = pd.read_csv(f"{DATA_VOLUME_MOUNTPATH}/{test_id}/{submission_id}/event_logs/TrainMovementEvents.trains_arrived.tsv", sep="\t")
  print(df)
  assert len(df) == 1
  print(df.iloc[0])
  success_rate = df.iloc[0]["success_rate"]
  print(success_rate)

  df = pd.read_csv(f"{DATA_VOLUME_MOUNTPATH}/{test_id}/{submission_id}/event_logs/TrainMovementEvents.trains_rewards_dones_infos.tsv", sep="\t")
  print(df)


  # TODO extract UUIDs and results from tsv
  results = [
    ('1ae61e4f-201b-4e97-a399-5c33fb75c57e', '557d9a00-7e6d-410b-9bca-a017ca7fe3aa', 'db5eaa85-3304-4804-b76f-14d23adb5d4c', 'primary', 100),
    ('1ae61e4f-201b-4e97-a399-5c33fb75c57e', '557d9a00-7e6d-410b-9bca-a017ca7fe3aa', 'db5eaa85-3304-4804-b76f-14d23adb5d4c', 'secondary', 1.0),
    ('564ebb54-48f0-4837-8066-b10bb832af9d', '557d9a00-7e6d-410b-9bca-a017ca7fe3aa', 'db5eaa85-3304-4804-b76f-14d23adb5d4c', 'primary', 100),
    ('564ebb54-48f0-4837-8066-b10bb832af9d', '557d9a00-7e6d-410b-9bca-a017ca7fe3aa', 'db5eaa85-3304-4804-b76f-14d23adb5d4c', 'secondary', 0.8)
  ]

  return results
