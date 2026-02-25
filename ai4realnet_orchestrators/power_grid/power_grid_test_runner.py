from abc import abstractmethod
import os
import requests
import tempfile
import zipfile

from lightsim2grid import LightSimBackend
import grid2op

from ai4realnet_orchestrators.test_runner import TestRunner


class PowerGridTestRunner(TestRunner):

  def init(self, submission_data_url: str, submission_id: str):
    super().init(submission_data_url=submission_data_url, submission_id=submission_id)
    self.submission_data = PowerGridTestRunner.load_submission_data(submission_data_url)

  def run_scenario(self, scenario_id: str, submission_id: str):
    scenario_data = self.submission_data[scenario_id]

    # Create environment
    env = grid2op.make(scenario_data['scenario_path'], backend=LightSimBackend())

    # Create and load agent
    agent = self.load_agent(scenario_data['agent_type'], scenario_data['agent_path'], env)

    return self.getResult(env, agent)

  @staticmethod
  def load_submission_data(submission_data_url: str) -> dict:
    response = requests.get(submission_data_url)
    return response.json()

  @staticmethod
  def load_agent(agent_type: str, agent_path: str | None, env):
    if agent_type == 'RandomAgent':
      from grid2op.Agent import RandomAgent
      agent = RandomAgent(env.action_space)

    elif agent_type == 'CurriculumAgent':
      # Create agent
      from ai4realnet_orchestrators.power_grid.framework.modified_curriculum_classes.baseline import CurriculumAgent
      agent = CurriculumAgent(env.action_space, env.observation_space, 'curriculum_agent')

      if agent_path is not None:
        # Extract agent zip locally
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(agent_path, 'r') as zip_ref:
          zip_ref.extractall(temp_dir)

        # Load model and actions
        model_path = os.path.join(temp_dir, 'model')
        actions_path = os.path.join(temp_dir, 'actions')
        agent.load(model_path, actions_path, best_action_threshold=0.95)

    else:
      raise SyntaxError(f'Unsupported agent type: {agent_type}')

    return agent

  @abstractmethod
  def getResult(self, env, agent) -> dict:
    raise NotImplementedError("This method should be overridden by subclasses.")
