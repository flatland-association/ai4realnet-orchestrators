import json
import logging
import os
import ssl
import subprocess
import time
import uuid
from io import TextIOWrapper, BytesIO
from typing import List, Optional

import pytest
from celery import Celery
from testcontainers.compose import DockerCompose

from ai4realnet_orchestrators.fab_oauth_utils import backend_application_flow
from fab_clientlib import DefaultApi, Configuration, ApiClient

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def test_containers_fixture():
  # set env var ATTENDED to True if docker-demo.yml is already up and running
  if os.environ.get("ATTENDED", "False").lower() == "true":
    yield
    return

  global basic

  start_time = time.time()
  basic = DockerCompose(context="../..", profiles=["full"])
  logger.info("/ start docker compose down")
  basic.stop()
  duration = time.time() - start_time
  logger.info(f"\\ end docker compose down. Took {duration:.2f} seconds.")
  start_time = time.time()
  logger.info("/ start docker compose up")
  try:
    basic.start()
    duration = time.time() - start_time
    logger.info(f"\\ end docker compose up. Took {duration:.2f} seconds.")

    submission_id = str(uuid.uuid4())
    yield submission_id

    # TODO workaround for testcontainers not supporting streaming to logger
    start_time = time.time()
    logger.info("/ start get docker compose logs")
    stdout, stderr = basic.get_logs()
    logger.info("stdout from docker compose")
    logger.info(stdout)
    logger.warning("stderr from docker compose")
    logger.warning(stderr)
    duration = time.time() - start_time
    logger.info(f"\\ end get docker compose logs. Took {duration:.2f} seconds.")

    start_time = time.time()
    logger.info("/ start docker compose down")
    basic.stop()
    duration = time.time() - start_time
    logger.info(f"\\ end docker down. Took {duration:.2f} seconds.")
  except BaseException as e:
    print("An exception occurred during running docker compose:")
    print(e)
    stdout, stderr = basic.get_logs()
    print(stdout)
    print(stderr)
    raise e


def run_task(task_queue_name: str, submission_id: str, submission_data_url: str, tests: List[str], **kwargs):
  start_time = time.time()
  app = Celery(
    broker="amqps://guest:guest@localhost:5671",
    backend="rpc://",
    broker_use_ssl={
      'keyfile': "../../docker/rabbitmq/certs/client_localhost_key.pem",
      'certfile': "../../docker/rabbitmq/certs/client_localhost_certificate.pem",
      'ca_certs': "../../docker/rabbitmq/certs/ca_certificate.pem",
      'cert_reqs': ssl.CERT_REQUIRED
    }
  )

  logger.info(f"/ Start simulate submission from portal for submission_id={submission_id}.....")

  ret = app.send_task(
    task_queue_name,
    task_id=submission_id,
    kwargs={
      "submission_data_url": submission_data_url,
      "tests": tests,
      **kwargs
    },
    queue=task_queue_name,
  ).get()
  logger.info(ret)
  duration = time.time() - start_time
  logger.info(
    f"\\ End simulate submission from portal for submission_id={submission_id}. Took {duration} seconds.")
  return ret


@pytest.mark.usefixtures("test_containers_fixture")
@pytest.mark.integration
def test_runner_kpi_pf_026_railway():
  task_queue_name = 'Railway'  # Celery: queue name = task name
  submission_id = str(uuid.uuid4())  # Celery: task ID
  test_id = "98ceb866-5479-47e6-a735-81292de8ca65"  # Celery: passed in "tests" key of kwargs when Celery task is submitted
  # TODO use versioned dependency instead of latest
  submission_data_url = "ghcr.io/flatland-association/flatland-baselines:latest"  # Celery: passed in "submission_data_url" key of kwargs when Celery task is submitted

  def _verify_kpi_pf_026(test_results):
    assert len(test_results.body) == 1
    test_results = test_results.body[0]
    assert test_results.scenario_scorings[0].scorings[0].field_key == "punctuality"
    assert test_results.scenario_scorings[0].scorings[0].score == 0.9285714285714286
    assert test_results.scenario_scorings[0].scorings[1].field_key == "success_rate"
    assert test_results.scenario_scorings[0].scorings[1].score == 1.0
    assert test_results.scenario_scorings[1].scorings[0].field_key == "punctuality"
    assert test_results.scenario_scorings[1].scorings[0].score == 1.0
    assert test_results.scenario_scorings[1].scorings[1].field_key == "success_rate"
    assert test_results.scenario_scorings[1].scorings[1].score == 1.0
    assert test_results.scorings[0].field_key == "punctuality"
    assert test_results.scorings[0].score == 0.9642857142857143

  _generic_run(submission_data_url, submission_id, task_queue_name, test_id, _verify_kpi_pf_026)


@pytest.mark.usefixtures("test_containers_fixture")
@pytest.mark.integration
def test_runner_kpi_nf_045_railway():
  task_queue_name = 'Railway'  # Celery: queue name = task name
  submission_id = str(uuid.uuid4())  # Celery: task ID
  test_id = "e075d4a7-5cda-4d3c-83ac-69a0db1d74dd"  # Celery: passed in "tests" key of kwargs when Celery task is submitted
  # TODO revert to latest once re-built
  submission_data_url = "ghcr.io/flatland-association/flatland-baselines:latest"  # Celery: passed in "submission_data_url" key of kwargs when Celery task is submitted

  def _verify_kpi_nf_045(test_results):
    assert len(test_results.body) == 1
    test_results = test_results.body[0]

    assert test_results.scenario_scorings[0].scorings[0].field_key == "network_impact_propagation"
    assert test_results.scenario_scorings[0].scorings[0].score == 0.8571428571428572
    assert test_results.scenario_scorings[0].scorings[1].field_key == "success_rate_1"
    assert test_results.scenario_scorings[0].scorings[1].score == 1.0
    assert test_results.scenario_scorings[0].scorings[2].field_key == "punctuality_1"
    assert test_results.scenario_scorings[0].scorings[2].score == 0.8571428571428571
    assert test_results.scenario_scorings[0].scorings[3].field_key == "success_rate_2"
    assert test_results.scenario_scorings[0].scorings[3].score == 1.0
    assert test_results.scenario_scorings[0].scorings[4].field_key == "punctuality_2"
    assert test_results.scenario_scorings[0].scorings[4].score == 0.8571428571428571

    assert test_results.scenario_scorings[1].scorings[0].score == 1
    assert test_results.scenario_scorings[1].scorings[1].field_key == "success_rate_1"
    assert test_results.scenario_scorings[1].scorings[1].score == 1.0
    assert test_results.scenario_scorings[1].scorings[2].field_key == "punctuality_1"
    assert test_results.scenario_scorings[1].scorings[2].score == 1
    assert test_results.scenario_scorings[1].scorings[3].field_key == "success_rate_2"
    assert test_results.scenario_scorings[1].scorings[3].score == 1.0
    assert test_results.scenario_scorings[1].scorings[4].field_key == "punctuality_2"
    assert test_results.scenario_scorings[1].scorings[4].score == 1

    assert test_results.scorings[0].field_key == "network_impact_propagation"
    assert test_results.scorings[0].score == 0.9285714285714286

  _generic_run(submission_data_url, submission_id, task_queue_name, test_id, _verify_kpi_nf_045)


def _generic_run(submission_data_url, submission_id, task_queue_name, test_id, verify):
  try:
    run_task(task_queue_name, submission_id, submission_data_url, tests=[test_id])

    token = backend_application_flow(
      client_id='fab-client-credentials',
      client_secret='top-secret',
      token_url='http://localhost:8081/realms/flatland/protocol/openid-connect/token',
    )
    print(token)
    fab = DefaultApi(ApiClient(configuration=Configuration(host="http://localhost:8000", access_token=token["access_token"])))

    test_results = fab.results_submissions_submission_id_tests_test_ids_get(
      submission_id=submission_id,
      test_ids=[test_id])
    print("results_uploaded")
    _pretty_print(test_results)
    verify(test_results)

  except BaseException as e:
    exec_with_logging(["docker", "ps"])
    debug = []
    try:
      logger.info("/ Logs from docker compose")

      stdo, stderr = exec_with_logging(["docker", "compose", "--profile", "full", "logs", ],
                                       log_level_stdout=logging.INFO,
                                       log_level_stderr=logging.WARN,
                                       collect=True)
      debug += stdo
      debug += stderr
      logger.info("\\ Logs from docker compose")
    except:
      logger.warning("Could not fetch logs from docker compose")
    raise Exception(str(e) + ": " + '\n'.join(debug)) from e


# https://stackoverflow.com/questions/21953835/run-subprocess-and-print-output-to-logging
def exec_with_logging(exec_args: List[str], log_level_stdout=logging.DEBUG, log_level_stderr=logging.WARN, collect: bool = False):
  logger.debug(f"/ Start %s", exec_args)
  try:
    proc = subprocess.Popen(exec_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = proc.communicate()
    stdo = log_subprocess_output(TextIOWrapper(BytesIO(stdout)), level=log_level_stdout, label=str(exec_args), collect=collect)
    stde = log_subprocess_output(TextIOWrapper(BytesIO(stderr)), level=log_level_stderr, label=str(exec_args), collect=collect)
    logger.debug("\\ End %s", exec_args)
    return stdo, stde
  except (OSError, subprocess.CalledProcessError) as exception:
    logger.error(stderr)
    raise RuntimeError(f"Failed to run {exec_args}. Stdout={stdout}. Stderr={stderr}") from exception


# https://stackoverflow.com/questions/21953835/run-subprocess-and-print-output-to-logging
def log_subprocess_output(pipe, level=logging.DEBUG, label="", collect: bool = False) -> Optional[List[str]]:
  s = []
  for line in pipe.readlines():
    logger.log(level, "[from subprocess %s] %s", label, line)
    if collect:
      s.append(line)
  if collect:
    return s
  return None


# https://stackoverflow.com/questions/36588126/uuid-is-not-json-serializable
class UUIDEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, uuid.UUID):
      # if the obj is uuid, we simply return the value of uuid
      return obj.hex
    return json.JSONEncoder.default(self, obj)


def _pretty_print(submissions):
  print(json.dumps(submissions.to_dict(), indent=4, cls=UUIDEncoder))
