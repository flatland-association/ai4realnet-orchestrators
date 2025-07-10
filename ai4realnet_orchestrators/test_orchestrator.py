import mockito
import pytest
from mockito import mock, when, verify

from ai4realnet_orchestrators.orchestrator import Orchestrator, TaskExecutionError
from ai4realnet_orchestrators.test_runner import TestRunner
from fab_clientlib import DefaultApi, ResultsSubmissionsSubmissionIdTestsTestIdsPostRequest, ResultsSubmissionsSubmissionIdTestsTestIdsPostRequestDataInner


def test_orchestrator():
  test_runner_dummy: TestRunner = mock()
  test_runner_yummy: TestRunner = mock()
  fab: DefaultApi = mock()
  orchestrator = Orchestrator(test_runners={"dummy": test_runner_dummy, "yummy": test_runner_yummy, })
  when(test_runner_dummy).run().thenReturn([("a", "primary", "55"), ("a", "secondary", "88")])

  when(test_runner_yummy).run().thenReturn([("f", "primary", "99"), ("f", "secondary", "101")])
  orchestrator.run("subi", "fancy", ["dummy", "yummy"], fab=fab)
  verify(test_runner_dummy, times=1).init(submission_data_url='fancy', submission_id='subi')
  verify(test_runner_yummy, times=1).init(submission_data_url='fancy', submission_id='subi')
  verify(fab, times=1).results_submissions_submission_id_tests_test_ids_post(
    submission_id="subi",
    test_ids=["dummy"],
    results_submissions_submission_id_tests_test_ids_post_request=ResultsSubmissionsSubmissionIdTestsTestIdsPostRequest(
      data=[
        ResultsSubmissionsSubmissionIdTestsTestIdsPostRequestDataInner(
          scenario_id="a",
          additional_properties={"primary": "55"},
        ),
        ResultsSubmissionsSubmissionIdTestsTestIdsPostRequestDataInner(
          scenario_id="a",
          additional_properties={"secondary": "88"},
        )
      ]
    )
  )
  verify(fab, times=1).results_submissions_submission_id_tests_test_ids_post(
    submission_id="subi",
    test_ids=["yummy"],
    results_submissions_submission_id_tests_test_ids_post_request=ResultsSubmissionsSubmissionIdTestsTestIdsPostRequest(
      data=[
        ResultsSubmissionsSubmissionIdTestsTestIdsPostRequestDataInner(
          scenario_id="f",
          additional_properties={"primary": "99"},
        ),
        ResultsSubmissionsSubmissionIdTestsTestIdsPostRequestDataInner(
          scenario_id="f",
          additional_properties={"secondary": "101"},
        )
      ]
    )
  )
  mockito.unstub()


def test_orchestrator_undefined_test():
  orchestrator = Orchestrator(test_runners={"it": TestRunner("it")})
  with pytest.raises(TaskExecutionError) as exc_info:
    orchestrator.run("subi", "fancy", ["else"])
  assert exc_info.value.status == {"orchestrator": "FAILED"}
  assert exc_info.value.message.startswith("Test else not implemented in <ai4realnet_orchestrators.orchestrator.Orchestrator")
