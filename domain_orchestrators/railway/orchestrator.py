# based on https://github.com/codalab/codabench/blob/develop/compute_worker/compute_worker.py
import logging
import os
import ssl
import traceback
from typing import Dict
from typing import List

from celery import Celery

from domain_orchestrators.orchestrator import Orchestrator
from domain_orchestrators.railway.test_runner_557d9a00 import TestRunner557d9a00
from fab_clientlib import DefaultApi, Configuration, ApiClient, ResultsSubmissionsSubmissionIdTestsTestIdsPostRequest, \
  ResultsSubmissionsSubmissionIdTestsTestIdsPostRequestDataInner

from fab_oauth_utils import backend_application_flow

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

# https://docs.celeryq.dev/en/stable/userguide/tasks.html#bound-tasks: A task being bound means the first argument to the task will always be the task instance (self).
# https://docs.celeryq.dev/en/stable/userguide/tasks.html#names: Every task must have a unique name.
# @app.task(name=os.environ.get("BENCHMARK_ID"), bind=True)
# def orchestrator(self, submission_data_url: str, tests: List[str] = None, **kwargs):
#
#
#   try:
#     # we use the submission_id as the unique id of the executing task.
#     submission_id = self.request.id
#     # we use the benchmark_id as the task's name and queue name (i.e. one task per benchmark). This ensures the Celery task is routed to the responsible orchestrator
#     benchmark_id = orchestrator.name
#     logger.info(f"Queue/task {benchmark_id} received submission {submission_id} with submission_data_url={submission_data_url} for tests={tests}")
#     for test_id in tests:
#       if test_id == "557d9a00-7e6d-410b-9bca-a017ca7fe3aa":
#         results = run_and_evaluate_test_557d9a00(submission_id=submission_id, test_id=test_id, submission_data_url=submission_data_url)
#       elif test_id == "[INSERT HERE: @TestId]":
#         pass
#       else:
#         raise TaskExecutionError(status={"orchestrator": "FAILED"}, message=f"Test {test_id} not implemented in {self}")
#
#       token = backend_application_flow(CLIENT_ID, CLIENT_SECRET, TOKEN_URL)
#       print(token)
#       print(results)
#       fab = DefaultApi(ApiClient(configuration=Configuration(host=FAB_API_URL, access_token=token["access_token"])))
#       fab.results_submissions_submission_id_tests_test_ids_post(
#         submission_id=submission_id,
#         test_ids=[test_id],
#         results_submissions_submission_id_tests_test_ids_post_request=ResultsSubmissionsSubmissionIdTestsTestIdsPostRequest(
#           data=[ResultsSubmissionsSubmissionIdTestsTestIdsPostRequestDataInner(
#             scenario_id=scenario_id,
#             additional_properties={key: value}
#           ) for scenario_id, key, value in results]
#         )
#       )
#     return {
#       "status": "SUCCESS",
#       "message": "message"
#     }
#   except BaseException as e:
#     raise Exception(f"{e} with tb {traceback.format_exc()}")
#
#


railway_orchestrator = Orchestrator(
  test_runners={
    "557d9a00-7e6d-410b-9bca-a017ca7fe3aa": TestRunner557d9a00(
      test_id="557d9a00-7e6d-410b-9bca-a017ca7fe3aa", scenario_ids=['1ae61e4f-201b-4e97-a399-5c33fb75c57e', '564ebb54-48f0-4837-8066-b10bb832af9d']
    ),
  }
)


@app.task(name=os.environ.get("BENCHMARK_ID"), bind=True)
def orchestrator(self, submission_data_url: str, tests: List[str] = None, **kwargs):
  submission_id = self.request.id
  benchmark_id = orchestrator.name
  logger.info(
    f"Queue/task {benchmark_id} received submission {submission_id} with submission_data_url={submission_data_url} for tests={tests}"
  )
  railway_orchestrator.run(
    submission_id=submission_id,
    submission_data_url=submission_data_url,
    tests=tests,
  )
