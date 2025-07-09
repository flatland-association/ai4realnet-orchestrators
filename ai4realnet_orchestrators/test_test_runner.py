import uuid

from ai4realnet_orchestrators.test_runner import TestRunner


class DummyTestRunner(TestRunner):
  def run_scenario(self, scenario_id: str, submission_id: str):
    return {
      "key_1": "value_1",
      "key_2": "value_2",
    }


def test_test_runner_run():
  test_id = uuid.uuid4()
  scenario1_id = str(uuid.uuid4())
  scenario2_id = str(uuid.uuid4())
  results = DummyTestRunner(
    test_id=str(test_id), scenario_ids=[scenario1_id, scenario2_id]
  ).run()
  assert results == [(scenario1_id, "key_1", "value_1"),
                     (scenario1_id, "key_2", "value_2"),
                     (scenario2_id, "key_1", "value_1"),
                     (scenario2_id, "key_2", "value_2"), ]
