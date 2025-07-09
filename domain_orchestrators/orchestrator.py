import os
import traceback
from typing import Dict, List

from domain_orchestrators.test_runner import TestRunner
from fab_clientlib import Configuration, ApiClient, DefaultApi, ResultsSubmissionsSubmissionIdTestsTestIdsPostRequestDataInner, \
  ResultsSubmissionsSubmissionIdTestsTestIdsPostRequest
from fab_oauth_utils import backend_application_flow

FAB_API_URL = os.environ.get("FAB_API_URL")
CLIENT_ID = os.environ.get("CLIENT_ID", 'fab-client-credentials')
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
TOKEN_URL = os.environ.get("TOKEN_URL", "https://keycloak.flatland.cloud/realms/flatland/protocol/openid-connect/token")


class TaskExecutionError(Exception):
  def __init__(self, message: str, status: Dict):
    super().__init__(message)
    self.message = message
    self.status = status


class Orchestrator:
  def __init__(self, test_runners: Dict[str, TestRunner] = None):
    self.test_runners = test_runners or {}

  def run(
    self, submission_id: str, submission_data_url: str, tests: List[str] = None
  ):
    """
    In general, no need to override.

    Parameters
    ----------
    submission_id : str
      passed for logging/temp file names.
    submission_data_url
      submission data url specifying the submission to be evaluated in the test.
    tests : List[str]
      singleton list containing test_id
    """
    try:
      for test_id in tests:
        test_runner = self.test_runners.get(tests[0])
        if not test_runner:
          raise TaskExecutionError(
            status={"orchestrator": "FAILED"},
            message=f"Test {test_id} not implemented in {self}",
          )
        test_runner.init(submission_data_url=submission_data_url, submission_id=submission_id)
        results = test_runner.run()

        token = backend_application_flow(CLIENT_ID, CLIENT_SECRET, TOKEN_URL)
        print(token)
        print(results)
        fab = DefaultApi(ApiClient(configuration=Configuration(host=FAB_API_URL, access_token=token["access_token"])))

        # could also be sent at once, but this way we get continuous updates
        fab.results_submissions_submission_id_tests_test_ids_post(
          submission_id=submission_id,
          test_ids=[test_id],
          results_submissions_submission_id_tests_test_ids_post_request=ResultsSubmissionsSubmissionIdTestsTestIdsPostRequest(
            data=[
              ResultsSubmissionsSubmissionIdTestsTestIdsPostRequestDataInner(
                scenario_id=scenario_id,
                additional_properties={key: value},
              )
              for scenario_id, key, value in results
            ]
          ),
        )
      return {"status": "SUCCESS", "message": f"Run submission {submission_id} for test {test_id} on submission data URL {submission_data_url}"}
    except BaseException as e:
      print(f"{e} with tb {traceback.format_exc()}")
      return {"status": "FAILED", "message": f"{e} with tb {traceback.format_exc()}"}
