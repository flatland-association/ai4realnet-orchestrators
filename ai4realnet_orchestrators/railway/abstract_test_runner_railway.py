import logging
import os
import shutil
from pathlib import Path
from typing import List

from ai4realnet_orchestrators.fab_exec_utils import exec_with_logging
from ai4realnet_orchestrators.s3_utils import s3_utils, S3_BUCKET
from ai4realnet_orchestrators.test_runner import TestRunner

# required only for docker in docker
DATA_VOLUME = os.environ.get("DATA_VOLUME")
SCENARIOS_VOLUME = os.environ.get("SCENARIOS_VOLUME")
SUDO = os.environ.get("SUDO", "true").lower() == "true"

DATA_VOLUME_MOUNTPATH = os.environ.get("DATA_VOLUME_MOUNTPATH", "/app/data")
SCENARIOS_VOLUME_MOUNTPATH = os.environ.get("SCENARIOS_VOLUME_MOUNTPATH", "/app/scenarios")
RAILWAY_ORCHESTRATOR_RUN_LOCAL = os.environ.get("RAILWAY_ORCHESTRATOR_RUN_LOCAL", False)

logger = logging.getLogger(__name__)


class AbtractTestRunnerRailway(TestRunner):
  def exec(self, generate_policy_args: List[str], scenario_id: str, submission_id: str, subdir: str):
    if not RAILWAY_ORCHESTRATOR_RUN_LOCAL:
      # --data-dir must exist -- TODO fix in flatland-rl instead
      args = ["docker", "run", "--rm", "-v", f"{DATA_VOLUME}:/vol", "alpine:latest", "mkdir", "-p", f"/vol/{subdir}"]
      exec_with_logging(args if not SUDO else ["sudo"] + args)
      args = ["docker", "run", "--rm", "-v", f"{DATA_VOLUME}:/vol", "alpine:latest", "chmod", "-R", "a=rwx",
              f"/vol/{submission_id}/{self.test_id}/{scenario_id}"]
      exec_with_logging(args if not SUDO else ["sudo"] + args)

      # update image
      args = ["docker", "pull", self.submission_data_url, ]
      exec_with_logging(args if not SUDO else ["sudo"] + args)
      args = [
               "docker", "run",
               "--rm",
               "-v", f"{DATA_VOLUME}:{DATA_VOLUME_MOUNTPATH}",
               "-v", f"{SCENARIOS_VOLUME}:{SCENARIOS_VOLUME_MOUNTPATH}",
               "--entrypoint", "/bin/bash",
               # Don't allow subprocesses to raise privileges, see https://github.com/codalab/codabench/blob/43e01d4bc3de26e8339ddb1463eef7d960ddb3af/compute_worker/compute_worker.py#L520
               "--security-opt=no-new-privileges",
               # Don't buffer python output, so we don't lose any
               "-e", "PYTHONUNBUFFERED=1",
               # for integration tests with localhost http
               "-e", "OAUTHLIB_INSECURE_TRANSPORT=1",
               self.submission_data_url,
               # TODO get rid of hard-coded path in flatland-baselines
               "/home/conda/entrypoint_generic.sh", "flatland-trajectory-generate-from-policy",
             ] + generate_policy_args
      exec_with_logging(args if not SUDO else ["sudo"] + args, log_level_stdout=logging.DEBUG)
    else:
      from flatland.trajectories.policy_runner import generate_trajectory_from_policy
      Path(f"{DATA_VOLUME_MOUNTPATH}/{subdir}").mkdir(parents=True, exist_ok=False)
      try:
        print(subdir)
        print(generate_policy_args)
        generate_trajectory_from_policy(generate_policy_args)
      except SystemExit as e_info:
        if e_info.code != 0:
          print(e_info)
        assert e_info.code == 0

  def upload_and_empty_local(self, submission_id: str, scenario_id: str):
    data_volume = Path(DATA_VOLUME)
    scenario_folder = data_volume / submission_id / self.test_id / scenario_id
    logger.info(f"Uploading {scenario_folder} to s3 {S3_BUCKET}/{scenario_folder.relative_to(data_volume)}")
    for f in scenario_folder.rglob("**/*"):
      s3_utils.upload_to_s3(f, f.relative_to(data_volume))
    logger.info(f"Deleting {scenario_folder} after uploading s3 {S3_BUCKET}/{scenario_folder.relative_to(data_volume)}")
    shutil.rmtree(scenario_folder)
