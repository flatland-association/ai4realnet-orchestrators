# based on https://github.com/codalab/codabench/blob/develop/compute_worker/compute_worker.py
import logging
import os
import ssl

from celery import Celery

from ai4realnet_orchestrators.atm.test_runner import BlueSkyRunner
from ai4realnet_orchestrators.orchestrator import Orchestrator

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

bluesky_orchestrator = Orchestrator(
    test_runners={
        "<test_id>": BlueSkyRunner(
            test_id="<test_id>", scenario_ids=['scnfile_name1', 'scnfile_name2']
        ),
    }
)


# https://docs.celeryq.dev/en/stable/userguide/tasks.html#bound-tasks: A task being bound means the first argument to the task will always be the task instance (self).
# https://docs.celeryq.dev/en/stable/userguide/tasks.html#names: Every task must have a unique name.
@app.task(name=os.environ.get("BENCHMARK_ID"), bind=True)
def orchestrator(self, submission_data_url: str, tests: list[str] = None, **kwargs):
    submission_id = self.request.id
    benchmark_id = orchestrator.name
    logger.info(
        f"Queue/task {benchmark_id} received submission {submission_id} with submission_data_url={submission_data_url} for tests={tests}"
    )
    return bluesky_orchestrator.run(
        submission_id=submission_id,
        submission_data_url=submission_data_url,
        tests=tests,
    )
