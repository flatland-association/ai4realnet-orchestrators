from typing import List


class TestRunner:
  def __init__(self, test_id: str, scenario_ids: List[str] = None):
    self.submission_data_url = None
    self.scenario_ids = scenario_ids or []
    self.test_id = test_id

  def init(self, submission_data_url: str):
    # if you want to run some initialization code, e.g. to load data from the submission_data_url
    self.submission_data_url = submission_data_url

    # TODO extract to template
    # submission_data = load_submission_data(submission_data_url)
    # self.model = load_model(submission_data)

  def run(self, submission_id: str):
    # override this method in case all scenarios should be run here, e.g. if special logic is needed
    results = []
    for scenario_id in self.scenario_ids:
      scenario_results = self.run_scenario(scenario_id, submission_id)
      for key, value in scenario_results.items():
        results.append((scenario_id, key, value))
    return results

  def run_scenario(self, scenario_id: str, submission_id: str):
    # this method should be overridden by subclasses
    # scenario_id is passed in case special data or logic is needed for a specific scenario
    raise NotImplementedError("This method should be overridden by subclasses.")

    # TODO extract to template
    # here you would implement the logic to run the test for the scenario:
    scenario_data = load_scenario_data(scenario_id)
    model = self.model

    # data and other stuff initialized in the init method can be used here
    # for demonstration, we return a dummy result
    return {
      "key_1": "value_1",
      "key_2": "value_2",
    }
