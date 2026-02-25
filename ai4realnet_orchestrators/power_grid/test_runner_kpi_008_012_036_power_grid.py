import json
import os
import numpy as np

from ai4realnet_orchestrators.power_grid.power_grid_test_runner import PowerGridTestRunner
from grid2op.utils import ScoreL2RPN2023
from grid2op.dtypes import dt_int

# TODO: check how this json was generated for all episodes in L2RPN
SCORING_CONFIG_JSON = './ai4realnet_orchestrators/power_grid/configuration/scoring-config.json'


def evaluate_kpis(env, agent, nb_scenario: int) -> dict:
  with open(SCORING_CONFIG_JSON) as config_file:
    config = json.load(config_file)

  nb_scenario = min(nb_scenario, int(config["nb_scenario"]))

  # env seeds are read from the config
  episodes_info = config["episodes_info"]
  env_seeds = [int(episodes_info[os.path.split(el)[-1]]["seed"])
               for el in sorted(env.chronics_handler.real_data.subpaths)
               if os.path.split(el)[-1] in episodes_info.keys()]

  # agent seeds are generated with the provided random seed
  np.random.seed(int(config["score_config"]["seed"]))
  max_int = np.iinfo(dt_int).max
  agent_seeds = list(np.random.randint(max_int, size=int(config["nb_scenario"])))

  scoring = ScoreL2RPN2023(env=env,
                          env_seeds=env_seeds[:nb_scenario],
                          agent_seeds=agent_seeds[:nb_scenario],
                          nb_scenario=nb_scenario,
                          min_losses_ratio=config["score_config"]["min_losses_ratio"],
                          verbose=0,
                          max_step=-1,
                          nb_process_stats=1,
                          add_nb_highres_sim=True,
                          weight_op_score=0.6,
                          weight_assistant_score=0.25,
                          weight_nres_score=0.15,
                          min_nres_score=-100,
                          min_assistant_score=-300)

  all_scores, _, _, _ = scoring.get(agent)
  scores_per_episode = {
    "op_score": [float(score[1]) for score in all_scores],
    "nres_score": [float(score[2]) for score in all_scores],
    "assistant_confidence_score": [float(score[3]) for score in all_scores],
  }

  episode_names = episodes_info.keys()
  score_config = config["score_config"]
  weights = [
    float(episodes_info[ep]["length"]) / float(score_config["total_timesteps"])
    for ep in sorted(episode_names)
  ]
  total_op_score = sum(w * s for w, s in zip(weights, scores_per_episode["op_score"]))
  total_nres_score = sum(w * s for w, s in zip(weights, scores_per_episode["nres_score"]))
  total_assistant_score = sum(w * s for w, s in zip(weights, scores_per_episode["assistant_confidence_score"]))

  return {
    "op_score": total_op_score,
    "nres_score": total_nres_score,
    "assistant_confidence_score": total_assistant_score,
  }


class TestRunner_KPI_AF_008_Power_Grid(PowerGridTestRunner):
  def getResult(self, env, agent) -> dict:
    scores = evaluate_kpis(env, agent, 2)
    return {
      "primary": scores["assistant_confidence_score"]
    }


class TestRunner_KPI_CF_012_Power_Grid(PowerGridTestRunner):
  def getResult(self, env, agent) -> dict:
    scores = evaluate_kpis(env, agent, 2)
    return {
      "primary": scores["nres_score"]
    }


class TestRunner_KPI_OF_036_Power_Grid(PowerGridTestRunner):
  def getResult(self, env, agent) -> dict:
    scores = evaluate_kpis(env, agent, 2)
    return {
      "primary": scores["op_score"]
    }
