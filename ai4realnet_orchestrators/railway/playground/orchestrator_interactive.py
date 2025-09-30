# based on https://github.com/codalab/codabench/blob/develop/compute_worker/compute_worker.py
import logging
import os
import ssl
from typing import List

from celery import Celery

from ai4realnet_orchestrators.orchestrator import Orchestrator
from ai4realnet_orchestrators.railway.playground.test_runner_playground_interactive import TestRunnerPlaygroundInteractive
from fab_clientlib import ApiClient, DefaultApi, Configuration

logger = logging.getLogger(__name__)

app = Celery(
  broker=os.environ.get('BROKER_URL'),
  backend=os.environ.get('BACKEND_URL'),
  queue=os.environ.get("BENCHMARK_ID"),
  broker_use_ssl={
    'keyfile': os.environ.get("RABBITMQ_KEYFILE"),
    'certfile': os.environ.get("RABBITMQ_CERTFILE"),
    'ca_certs': os.environ.get("RABBITMQ_CA_CERTS"),
    'cert_reqs': ssl.CERT_REQUIRED
  }
)

# Playground: https://ai4realnet-int.flatland.cloud/benchmarks/9fbde927-189e-44bb-8432-f63b491aabb0/734144d1-c88c-4371-9dcf-72dd5dfe058e
interactive_railway_orchestrator = Orchestrator(
  test_runners={
    "c4c70f8a-679c-4044-a9d4-5e0ce0780a0f": TestRunnerPlaygroundInteractive(
      test_id="c4c70f8a-679c-4044-a9d4-5e0ce0780a0f", scenario_ids=['cf8e0a9b-14af-43e1-b1fc-e43bf3aaddd7']
    )
  }
)


# https://docs.celeryq.dev/en/stable/userguide/tasks.html#bound-tasks: A task being bound means the first argument to the task will always be the task instance (self).
# https://docs.celeryq.dev/en/stable/userguide/tasks.html#names: Every task must have a unique name.
@app.task(name=os.environ.get("BENCHMARK_ID"), bind=True)
def orchestrator(self, submission_data_url: str, tests: List[str] = None, **kwargs):
  submission_id = self.request.id
  benchmark_id = orchestrator.name
  logger.info(
    f"Queue/task {benchmark_id} received submission {submission_id} with submission_data_url={submission_data_url} for tests={tests}"
  )
  # fail fast
  check_fab_healthy()
  return interactive_railway_orchestrator.run(
    submission_id=submission_id,
    submission_data_url=submission_data_url,
    tests=tests,
  )


def check_fab_healthy():
  FAB_API_URL = os.environ.get("FAB_API_URL")
  fab = DefaultApi(ApiClient(configuration=Configuration(host=FAB_API_URL)))
  print(fab.health_live_get())
