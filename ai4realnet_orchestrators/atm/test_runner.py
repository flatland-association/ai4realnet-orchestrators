import os
import pathlib
import pandas as pd

from ai4realnet_orchestrators.fab_exec_utils import exec_with_logging
from ai4realnet_orchestrators.test_runner import TestRunner

WORKDIR = os.environ.get("WORKDIR", "/data")


class BlueSkyRunner(TestRunner):
    def run_scenario(self, scenario_id: str, submission_id: str):
        # here you would implement the logic to run the test for the scenario:
        args = ["bluesky", "--detached", "--workdir", WORKDIR, "--scenfile", scenario_id]
        exec_with_logging(args)
        
        # Read the generated data file
        files = list(pathlib.Path(WORKDIR).glob("*.csv"))
        latest = max(files, key=os.path.getctime)
        data = pd.read_csv(latest, comment="#")
        return {
            "output": data,
        }
