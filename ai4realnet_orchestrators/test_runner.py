from abc import abstractmethod
from typing import List


class TestRunner:
  def __init__(self, test_id: str, scenario_ids: List[str] = None):
    self.submission_id = None
    self.submission_data_url = None
    self.scenario_ids = scenario_ids or []
    self.test_id = test_id

  def init(self, submission_data_url: str, submission_id: str):
    """
    Override to run some initialization code, e.g. to load data from the `submission_data_url`

    Parameters
    ----------
    submission_data_url : str
      submission data url specifying the submission to be evaluated in the test.
    submission_id : str
      passed for logging/temp file names.
    """
    self.submission_data_url = submission_data_url
    self.submission_id = submission_id

  def run(self):
    # override this method in case all scenarios should be run here, e.g. if special logic is needed
    results = []
    for scenario_id in self.scenario_ids:
      scenario_results = self.run_scenario(scenario_id, self.submission_id)
      for key, value in scenario_results.items():
        results.append((scenario_id, key, value))
    return results

  @abstractmethod
  def run_scenario(self, scenario_id: str, submission_id: str):
    """
    This method must be overridden by subclasses
    scenario_id

    Parameters
    ----------
    scenario_id : str
      passed in case special data or logic is needed for a specific scenario
    submission_id
      passed for logging or temp file names
    Returns
    -------

    """
    raise NotImplementedError("This method should be overridden by subclasses.")
