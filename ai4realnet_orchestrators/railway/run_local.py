import logging
import os
import uuid

from ai4realnet_orchestrators.railway.test_runner_kpi_pf_026_railway import DATA_VOLUME_MOUNTPATH
from ai4realnet_orchestrators.railway.test_submission import _get_test_runner


# requires env with railway orchestrator dependencies and flatland-baselines installed
def run_local(
  test
):
  logging.basicConfig(level=logging.INFO)
  assert os.environ.get("RAILWAY_ORCHESTRATOR_RUN_LOCAL")

  test_runner = _get_test_runner(test)
  test_id = test_runner.test_id
  submission_id = uuid.uuid4()

  print(f"{DATA_VOLUME_MOUNTPATH}/{submission_id}/{test_id}/")
  # TODO use versioned dependency instead of latest
  test_runner.init(submission_data_url="ghcr.io/flatland-association/flatland-baselines:latest", submission_id=submission_id)
  test_runner.run()


if __name__ == '__main__':
  run_local(
    # test="KPI-PF-026"  # Punctuality,
    test="KPI-NF-045"  # Network Impact Propagation,
  )
