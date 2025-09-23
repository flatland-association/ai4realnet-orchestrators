import logging
import os
from typing import List

from ai4realnet_orchestrators.fab_exec_utils import exec_with_logging
from ai4realnet_orchestrators.test_runner import TestRunner

# required only for docker in docker
DATA_VOLUME = os.environ.get("DATA_VOLUME")
SCENARIOS_VOLUME = os.environ.get("SCENARIOS_VOLUME")
SUDO = os.environ.get("SUDO", "true").lower() == "true"

DATA_VOLUME_MOUNTPATH = os.environ.get("DATA_VOLUME_MOUNTPATH", "/app/data")
SCENARIOS_VOLUME_MOUNTPATH = os.environ.get("SCENARIOS_VOLUME_MOUNTPATH", "/app/scenarios")


class AbtractTestRunnerRailway(TestRunner):
  def exec(self, generate_policy_args: List[str], scenario_id: str, submission_id: str):
    if True:
      # --data-dir must exist -- TODO fix in flatland-rl instead
      args = ["docker", "run", "--rm", "-v", f"{DATA_VOLUME}:/vol", "alpine:latest", "mkdir", "-p", f"/vol/{submission_id}/{self.test_id}/{scenario_id}"]
      exec_with_logging(args if not SUDO else ["sudo"] + args)
      args = ["docker", "run", "--rm", "-v", f"{DATA_VOLUME}:/vol", "alpine:latest", "chmod", "-R", "a=rwx",
              f"/vol/{submission_id}/{self.test_id}/{scenario_id}"]
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
      generate_trajectory_from_policy(generate_policy_args)
