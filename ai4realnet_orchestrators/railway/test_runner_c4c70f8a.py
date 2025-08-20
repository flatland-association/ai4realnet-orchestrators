from ai4realnet_orchestrators.test_runner import TestRunner


class TestRunnerc4c70f8a(TestRunner):
    def run_scenario(self, scenario_id: str, submission_id: str):
        seed = TestRunnerc4c70f8a.load_scenario_data(scenario_id)

        # TODO run olten with seed


        return {
            'primary': -1
        }

    @staticmethod
    def load_scenario_data(scenario_id: str) -> str:
        return {'cf8e0a9b-14af-43e1-b1fc-e43bf3aaddd7': "42"}[scenario_id]
