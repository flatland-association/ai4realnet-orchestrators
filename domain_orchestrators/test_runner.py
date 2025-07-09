from typing import List


class TestRunner:
  def __init__(self, test_id: str, scenario_ids: List[str] = None):
    self.submission_data_url = None
    self.scenario_ids = scenario_ids or []
    self.test_id = test_id

  def init(self, submission_data_url: str):
    """
    To run some initialization code, e.g. to load data from the submission_data_url

    Parameters
    ----------
    submission_data_url : str
    """
    self.submission_data_url = submission_data_url

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
