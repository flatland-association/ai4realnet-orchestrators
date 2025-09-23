from ai4realnet_orchestrators.railway.abstract_test_runner_railway import AbtractTestRunnerRailway


# KPI-AF-029: AI Response time (Railway)
class TestRunner_KPI_AF_029_Railway(AbtractTestRunnerRailway):

  def run_scenario(self, scenario_id: str, submission_id: str):
    raise NotImplementedError()

  @staticmethod
  def load_scenario_data(scenario_id: str) -> str:
    return {'c5219c2e-c3b9-4e7a-aefc-b767a9b3005d': "42"}[scenario_id]
