"""
Power Grid Test Runner - Combined KPI Implementations
======================================================

This module provides the base TestRunner for Power Grid KPIs and implementations
for multiple KPI categories.

Base Classes:
    - PowerGridTestRunner: Template for all Power Grid KPIs
    - OperationalTestRunner: Generic runner for Operational KPIs (008, 012, 036)
    - RobustnessResilienceTestRunner: Generic runner for Robustness/Resilience KPIs (069-077)
    - ReliabilityTestRunner: Generic runner for Reliability KPIs (052, 057)

Operational KPIs (KPI-008, KPI-012, KPI-036):
    - TestRunner_KPI_AF_008_Power_Grid: Assistant alert accuracy
    - TestRunner_KPI_CF_012_Power_Grid: Carbon intensity  
    - TestRunner_KPI_OF_036_Power_Grid: Operation score

Robustness KPIs (KPI-069 to KPI-073):
    - TestRunner_KPI_DF_069_Power_Grid: Drop-off in reward
    - TestRunner_KPI_FF_070_Power_Grid: Frequency changed output AI agent
    - TestRunner_KPI_SF_071_Power_Grid: Severity of changed output AI agent
    - TestRunner_KPI_SF_072_Power_Grid: Steps survived with perturbations
    - TestRunner_KPI_VF_073_Power_Grid: Vulnerability to perturbation

Resilience KPIs (KPI-074 to KPI-077):
    - TestRunner_KPI_AF_074_Power_Grid: Area between reward curves
    - TestRunner_KPI_DF_075_Power_Grid: Degradation time
    - TestRunner_KPI_RF_076_Power_Grid: Restorative time
    - TestRunner_KPI_SF_077_Power_Grid: Similarity state to unperturbed situation

Reliability KPIs (KPI-052, KPI-057):
    - TestRunner_KPI_DF_052_Power_Grid: Domain shift adaptation time
    - TestRunner_KPI_DF_057_Power_Grid: Domain shift success rate drop (performance drop)

Evaluation Caching:
    Each category of KPIs (Operational, Reliability, Robustness/Resilience)
    runs a single comprehensive evaluation that computes multiple metrics.
    Results are cached by `submission_id` within each runner class to ensure
    efficiency when multiple KPIs from the same category are evaluated.

Authors: INESC TEC (Robustness/Resilience), AI4REALNET Consortium
"""
import json
import random
from abc import abstractmethod
import logging
import os
import pickle

import pandas as pd
import requests
import sys
import tempfile
import zipfile
from typing import Dict, List

import numpy as np
from lightsim2grid import LightSimBackend
import grid2op
from grid2op.utils import ScoreL2RPN2023
from domain_shift_kpis.adaptation_time import DsAdaptationTime

from ai4realnet_orchestrators.test_runner import TestRunner

logger = logging.getLogger(__name__)

# ============================================================================
# Path Setup for Framework
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_PATH = os.path.join(SCRIPT_DIR, "framework")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "configuration")

# Add framework to path for imports
if FRAMEWORK_PATH not in sys.path:
    sys.path.insert(0, FRAMEWORK_PATH)


# ============================================================================
# Base Class: PowerGridTestRunner (TEMPLATE - DO NOT MODIFY)
# ============================================================================

class PowerGridTestRunner(TestRunner):
    """
    Base TestRunner for Power Grid KPIs.

    Provides common functionality:
    - Agent loading (CurriculumAgent, RandomAgent)
    - Environment creation with LightSimBackend
    - Submission data handling

    Subclasses must implement: getResult(env, agent) -> dict
    """

    def init(self, submission_data_url: str, submission_id: str):
        """Initialize and load submission data from JSON URL."""
        super().init(submission_data_url=submission_data_url, submission_id=submission_id)
        self.submission_data = PowerGridTestRunner.load_submission_data(submission_data_url)

    def run_scenario(self, scenario_id: str, submission_id: str):
        """Run evaluation for a specific scenario."""
        if scenario_id not in self.submission_data["scenarios"]:
            raise ValueError(f"Unrecognized scenario ID: '{scenario_id}'")

        default_config = self.submission_data["default_config"]
        specific_config = self.submission_data["specific_config"].get(scenario_id, {})
        # Merge default and specific configs (specific overrides default)
        scenario_data = {**default_config, **specific_config}

        # Load mapping towards scenario and agent paths
        with open(os.path.join(CONFIG_PATH, "path-mapping.json"), "r") as f:
            mapping = json.load(f)

        # Create environment with fast backend
        scenario_name = scenario_data["scenario_name"]
        scenario_path = mapping["scenario_path"][scenario_name]
        env = grid2op.make(scenario_path, backend=LightSimBackend())

        # Create shift environment when provided
        # TODO: update scenario_shift_path in path-mapping.json, it's the same as scenario_path for now
        scenario_shift_name = scenario_data.get("scenario_shift_name")
        if scenario_shift_name:
            scenario_shift_path = mapping["scenario_path"][scenario_shift_name]
            env_shift = grid2op.make(scenario_shift_path, backend=LightSimBackend())
        else:
            env_shift = None

        # Create and load agent
        agent_type = scenario_data["agent_type"]
        agent_path = mapping["agent_path"][agent_type]
        agent = self.load_agent(agent_type, agent_path, env)

        return self.getResult(env, env_shift, agent)

    @staticmethod
    def load_submission_data(submission_data_url: str) -> dict:
        """Load submission metadata from URL."""
        response = requests.get(submission_data_url)
        return response.json()

    @staticmethod
    def load_agent(agent_type: str, agent_path: str | None, env):
        """
        Load agent based on type.

        Supported types:
        - RandomAgent: Grid2Op random agent
        - CurriculumAgent: Trained curriculum learning agent
        - ExpertAgent: Trained expert learning agent
        """
        if agent_type == 'RandomAgent':
            from grid2op.Agent import RandomAgent
            agent = RandomAgent(env.action_space)

        elif agent_type == 'CurriculumAgent':
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
                
        elif agent_type == "ExpertAgent":
            from ExpertAgent.utils.helper_functions import make_gymenv
            from ExpertAgent.ExpertAgent import ExpertAgentRL
            from stable_baselines3.ppo import MlpPolicy
            
            env_gym = make_gymenv(env, obs_attr_to_keep=["rho"], action_space_path="read_from_file", act_to_keep=("set_bus",))
            model_path = agent_path
            nn_kwargs = {
                "policy": MlpPolicy,
                "env": env_gym,
                "verbose": True,
                "learning_rate": 1e-3,
                "tensorboard_log": model_path,
                "policy_kwargs": {"net_arch": [800, 1000, 1000, 800]},
                "device": "auto"
            }
            agent = ExpertAgentRL(name="PPO_SB3",
                                  env=env,
                                  action_space=env.action_space,
                                  gymenv=env_gym,
                                  gym_act_space=env_gym.action_space,
                                  gym_obs_space=env_gym.observation_space,
                                  nn_kwargs=nn_kwargs
                                  )
            agent.load(model_path)
        else:
            raise SyntaxError(f'Unsupported agent type: {agent_type}')

        return agent

    @abstractmethod
    def _compute_all_metrics(self, env, env_shift, agent) -> Dict:
        """
        Compute all metrics for this category of KPIs.

        Must be implemented by generic subclasses (Operational, Reliability, etc.).

        Args:
            env: Grid2Op environment
            env_shift: Grid2Op shift environment (if applicable)
            agent: Loaded agent

        Returns:
            dict containing all computed metrics
        """
        pass

    def getResult(self, env, env_shift, agent) -> dict:
        """
        Compute and return KPI results using caching.

        This generic implementation handles caching results by submission_id
        and extracting the specific metric for the current KPI.

        Args:
            env: Grid2Op environment
            env_shift: Grid2Op shift environment
            agent: Loaded agent

        Returns:
            dict with "primary" key containing the KPI value
        """
        # Use submission_id for caching
        cache_key = getattr(self, 'submission_id', 'default')

        # Get KPI info (set by subclasses)
        kpi_info = getattr(self, 'kpi_info', {
            "name": "Unknown KPI",
            "metric_key": None,
            "description": ""
        })

        logger.info(
            f"Running evaluation for {kpi_info['name']}\n"
            f"  Cache key: {cache_key}"
        )

        # Check cache - avoid re-running expensive evaluation
        # Subclasses should define their own _metrics_cache class attribute
        cache = getattr(self, '_metrics_cache', None)
        if cache is None:
            # Fallback to instance cache if class cache not defined
            if not hasattr(self, '_instance_metrics_cache'):
                self._instance_metrics_cache = {}
            cache = self._instance_metrics_cache

        if cache_key not in cache:
            logger.info(f"Cache miss - running complete evaluation in {self.__class__.__name__}")
            all_metrics = self._compute_all_metrics(env, env_shift, agent)
            cache[cache_key] = all_metrics
            logger.info(f"Cached results for {cache_key}")
        else:
            logger.info(f"Cache hit - using existing results for {cache_key}")

        all_metrics = cache[cache_key]

        # Extract KPI-specific value
        metric_key = kpi_info.get('metric_key')
        if metric_key is None:
            logger.error(f"No metric_key defined for test_id {getattr(self, 'test_id', 'unknown')}")
            return {"primary": 0.0}

        kpi_value = all_metrics.get(metric_key, 0.0)

        logger.info(
            f"KPI Result: {kpi_info['name']} = {kpi_value}\n"
            f"  Description: {kpi_info['description']}"
        )

        return {"primary": float(kpi_value)}


# ============================================================================
# Operational KPIs (008, 012, 036) - Using ScoreL2RPN2023
# ============================================================================

def get_scoring_config(chronics_path: str, seed: int):
    """
    Retrieve scenario information (number, length) for operational KPI calculation and generate seeds.
    """
    scenario_names = sorted([name for name in os.listdir(chronics_path)
                             if os.path.isdir(os.path.join(chronics_path, name))])
    scenario_lengths = [len(pd.read_csv(os.path.join(chronics_path, name + "/load_p.csv.bz2"), compression="bz2"))
                        for name in scenario_names]

    rng = random.Random(seed)
    config = {
        "nb_scenario": len(scenario_names),
        "total_timesteps": sum(scenario_lengths),
        "episodes_info": {}
    }

    for name, length in zip(scenario_names, scenario_lengths):
        config["episodes_info"][name] = {
            "length": length,
            "env_seed": rng.randint(0, 2**31 - 1),
            "agent_seed": rng.randint(0, 2**31 - 1)
        }

    return config


def evaluate_operational_kpis(env, agent, nb_scenario: int = 9999) -> dict:
    """
    Evaluate operational KPIs using Grid2Op's ScoreL2RPN2023.

    Returns scores for:
    - op_score: Operation score
    - nres_score: Non-renewable energy score (carbon intensity)
    - assistant_confidence_score: Assistant alert accuracy
    """
    chronics_path = os.path.join(env.get_path_env(), "chronics")
    config = get_scoring_config(chronics_path, 4295) # fixed seed for reproducibility

    episodes_info = config["episodes_info"]
    scenario_names = episodes_info.keys()
    env_seeds = [int(episodes_info[name]["env_seed"]) for name in scenario_names]
    agent_seeds = [int(episodes_info[name]["agent_seed"]) for name in scenario_names]

    nb_scenario = min(nb_scenario, int(config["nb_scenario"]))
    scoring = ScoreL2RPN2023(
        env=env,
        env_seeds=env_seeds[:nb_scenario],
        agent_seeds=agent_seeds[:nb_scenario],
        nb_scenario=nb_scenario,
        min_losses_ratio=0.8,
        verbose=0,
        max_step=-1,
        nb_process_stats=1,
        add_nb_highres_sim=True,
        weight_op_score=0.6,
        weight_assistant_score=0.25,
        weight_nres_score=0.15,
        min_nres_score=-100,
        min_assistant_score=-300
    )

    all_scores, _, _, _ = scoring.get(agent)
    scores_per_episode = {
        "op_score": [float(score[1]) for score in all_scores],
        "nres_score": [float(score[2]) for score in all_scores],
        "assistant_confidence_score": [float(score[3]) for score in all_scores],
    }

    weights = [float(episodes_info[name]["length"]) / float(config["total_timesteps"]) for name in scenario_names]
    total_op_score = sum(w * s for w, s in zip(weights, scores_per_episode["op_score"]))
    total_nres_score = sum(w * s for w, s in zip(weights, scores_per_episode["nres_score"]))
    total_assistant_score = sum(w * s for w, s in zip(weights, scores_per_episode["assistant_confidence_score"]))

    return {
        "op_score": total_op_score,
        "nres_score": total_nres_score,
        "assistant_confidence_score": total_assistant_score,
    }


# KPI ID to metric mapping for Operational KPIs
OPERATIONAL_KPI_MAPPING = {
    # Operational KPIs (Benchmark: 4b0be731-8371-4e4e-a673-b630187b0bb8)
    "aba10b3f-0d5c-4f90-aec4-69460bbb098b": {
        "name": "KPI-AF-008: Assistant alert accuracy",
        "metric_key": "assistant_confidence_score",
        "description": "Assistant alert accuracy [0-100]"
    },
    "ab91af79-ffc3-4da7-916a-6574609dc1b6": {
        "name": "KPI-CF-012: Carbon intensity",
        "metric_key": "nres_score",
        "description": "Carbon intensity [0-100]"
    },
    "ae4dcac7-c559-457e-902d-ee35d064bb3f": {
        "name": "KPI-OF-036: Operation score",
        "metric_key": "op_score",
        "description": "Operation score [0-100]"
    }
}


class OperationalTestRunner(PowerGridTestRunner):
    """
    Extended TestRunner for Operational KPIs (008, 012, 036).

    Inherits from PowerGridTestRunner and implements getResult() to run
    operational evaluation. Results are cached per submission.
    """

    # Class-level cache: {submission_id: all_metrics_dict}
    _metrics_cache: Dict[str, Dict] = {}

    def __init__(self, test_id: str, scenario_ids: List[str], benchmark_id: str):
        """Initialize with KPI-specific configuration."""
        super().__init__(test_id=test_id, benchmark_id=benchmark_id)
        self.scenario_ids = scenario_ids

        # Get KPI info from mapping
        self.kpi_info = OPERATIONAL_KPI_MAPPING.get(test_id, {
            "name": "Unknown KPI",
            "metric_key": None,
            "description": ""
        })

        logger.info(
            f"Initialized OperationalTestRunner\n"
            f"  Test ID: {test_id}\n"
            f"  KPI: {self.kpi_info['name']}"
        )

    def _compute_all_metrics(self, env, env_shift, agent) -> Dict:
        """Evaluate operational KPIs and return all metrics."""
        return evaluate_operational_kpis(env, agent)


class TestRunner_KPI_AF_008_Power_Grid(OperationalTestRunner):
    """KPI-AF-008: Assistant alert accuracy"""
    pass


class TestRunner_KPI_CF_012_Power_Grid(OperationalTestRunner):
    """KPI-CF-012: Carbon intensity"""
    pass


class TestRunner_KPI_OF_036_Power_Grid(OperationalTestRunner):
    """KPI-OF-036: Operation score"""
    pass

# ============================================================================
# Reliability & Domain shift KPIs (052-058 + 090)
# ============================================================================

def evaluate_domain_shift_kpis(env, env_shift, agent) -> Dict:
    if env_shift is None:
        logger.warning("No shift environment provided. Domain shift KPIs cannot be computed.")
        return {
            "adaptation_time": 0.0,
            "performance_drop": 0.0
        }

    from ExpertAgent.utils.helper_functions import make_gymenv
    env_gym = make_gymenv(env, obs_attr_to_keep=["rho"], action_space_path="read_from_file", act_to_keep=("set_bus",))
    env_gym_shift = make_gymenv(env_shift, obs_attr_to_keep=["rho"], action_space_path="read_from_file", act_to_keep=("set_bus",))
    
    ds_kpi = DsAdaptationTime(agent=agent, 
                              trained_model_path=None, 
                              env=env_gym, 
                              env_shift=env_gym_shift
                             )
    
    # save_path = os.path.join(here, "..", "trained_models", "PPO_SB3_FINETUNE")
    
    train_kwargs = {
        "train_steps": int(1e3),
        "load_path": None,
        "save_path": None,
        "save_freq": 5000,
    }
    
    eval_kwargs = {
        "n_eval_episodes": 10,
        "render": False,
        "deterministic": True,
        "return_episode_rewards": True
    }
    
    results = ds_kpi.compute(acceptance_threshold=200.,
                             fine_tune_budget=int(15e3),
                             agent_train_fun=agent.train_static,
                             agent_train_kwargs=train_kwargs,
                             agent_eval_fun=agent.evaluate,
                             agent_eval_kwargs=eval_kwargs,
                             min_train_steps=int(1e3),
                             save_path=None
                             )
    return results

# KPI ID to metric mapping
RELIABILITY_KPI_MAPPING = {
    # Robustness KPIs (Benchmark: 3810191b-8cfd-4b03-86b2-f7e530aab30d)
    "855729a4-6729-4ae2-bb8d-443ef4867d94": {
        "name": "KPI-DF-052: Domain shift adaptation time",
        "metric_key": "adaptation_time",
        "description": "Iterations required for an agent to adapt its policy to domain shift"
    },
    "c5e4f893-4302-47e8-98d6-b5fbcb10963a": {
        "name": "KPI-DF-057: Domain shift success rate drop",
        "metric_key": "performance_drop",
        "description": "The performance drop when encountering a domain shift"
    },
}

class ReliabilityTestRunner(PowerGridTestRunner):
    """
    Extended TestRunner for Reliability KPIs (052-058 + 090).
    
    Inherits from PowerGridTestRunner and implements getResult() to run
    the reliability evaluation against a RL agent
    
    Single evaluation computes ALL the metrics
    """
    
    # Class-level cache: {submission_id: all_metrics_dict}
    _metrics_cache: Dict[str, Dict] = {}
    
    def __init__(self, test_id: str, scenario_ids: List[str], benchmark_id: str):
        """Initialize with KPI-specific configuration."""
        super().__init__(test_id=test_id, benchmark_id=benchmark_id)
        self.scenario_ids = scenario_ids
        
        # Get KPI info from mapping
        self.kpi_info = RELIABILITY_KPI_MAPPING.get(test_id, {
            "name": "Unknown KPI",
            "metric_key": None,
            "description": ""
        })
        
        logger.info(
            f"Initialized ReliabilityTestRunner\n"
            f"  Test ID: {test_id}\n"
            f"  KPI: {self.kpi_info['name']}"
        )
    
    def _compute_all_metrics(self, env, env_shift, agent) -> Dict:
        """Evaluate reliability KPIs and return all metrics."""
        return evaluate_domain_shift_kpis(env, env_shift, agent)


class TestRunner_KPI_DF_052_Power_Grid(ReliabilityTestRunner):
    """KPI-DF-052: Domain shift adaptation time"""
    pass


class TestRunner_KPI_DF_057_Power_Grid(ReliabilityTestRunner):
    """KPI-DF-057: Domain shift Success Rate Drop (performance drop)"""
    pass


# ============================================================================
# Robustness & Resilience KPIs (069-077) - Configuration
# ============================================================================

# KPI ID to metric mapping
ROBUSTNESS_RESILIENCE_KPI_MAPPING = {
    # Robustness KPIs (Benchmark: 3810191b-8cfd-4b03-86b2-f7e530aab30d)
    "1cbb7783-47b4-4289-9abf-27939da69a2f": {
        "name": "KPI-DF-069: Drop-off in reward",
        "metric_key": "reward_drop_percent",
        "description": "Percentage decrease in reward [0-100]"
    },
    "acaf712a-c06c-4a04-a00f-0e7feeefb60c": {
        "name": "KPI-FF-070: Frequency changed output",
        "metric_key": "action_change_freq",
        "description": "Proportion of timesteps with changed actions [0-1]"
    },
    "3d033ec6-942a-4b03-b26e-f8152ba48022": {
        "name": "KPI-SF-071: Severity of changed output",
        "metric_key": "severity_of_change",
        "description": "Severity of action changes [0-1, higher=worse]"
    },
    "a121d8bd-1943-41ba-b3a7-472a0154f8f9": {
        "name": "KPI-SF-072: Steps survived with perturbations",
        "metric_key": "n_steps_survived",
        "description": "Number of timesteps before failure"
    },
    "b8a9a411-7cfe-4c1d-b9a6-eef1c0efe920": {
        "name": "KPI-VF-073: Vulnerability to perturbation",
        "metric_key": "perturb_vulnerability",
        "description": "Proportion of features vulnerable to attack [0-1]"
    },
    # Resilience KPIs (Benchmark: 31ea606b-681a-437a-85b9-7c81d4ccc287)
    "534f5a1f-7115-48a5-b58c-4deb044d425d": {
        "name": "KPI-AF-074: Area between reward curves",
        "metric_key": "area_between_curves",
        "description": "Integrated performance degradation"
    },
    "04a23bfc-fc44-4ec4-a732-c29214130a83": {
        "name": "KPI-DF-075: Degradation time",
        "metric_key": "degradation_time",
        "description": "Time until performance degrades"
    },
    "225aaee8-7c7f-4faf-810b-407b551e9f2a": {
        "name": "KPI-RF-076: Restorative time",
        "metric_key": "restoration_time",
        "description": "Time to restore performance"
    },
    "7fe4210f-1253-411c-ba03-49d8b37c71fa": {
        "name": "KPI-SF-077: Similarity to unperturbed state",
        "metric_key": "state_similarity",
        "description": "Cosine similarity to unperturbed states [-1 to 1]"
    },
}


# ============================================================================
# Robustness & Resilience Base Class
# ============================================================================

class RobustnessResilienceTestRunner(PowerGridTestRunner):
    """
    Extended TestRunner for Robustness & Resilience KPIs (069-077).
    
    Inherits from PowerGridTestRunner and implements getResult() to run
    multi-attacker evaluation against the defender agent.
    
    Attackers:
    - GEPerturb: Gradient estimation perturbation
    - LambdaPIR: Lambda policy iteration with refinement
    - Random: Random perturbations
    - PPO: PPO-trained attacker
    - SAC_5/SAC_10: SAC-trained attackers with different factors
    - RLPerturb: RL-based perturbation agent
    
    Single evaluation computes ALL 9 metrics, results are cached to avoid
    re-running when multiple KPIs are evaluated for the same submission.
    """
    
    # Evaluation configuration
    ATTACKER_TYPES = ["GEPerturb", "LambdaPIR", "Random", "PPO", "SAC_10", "SAC_5", "RLPerturb"]
    NUM_EPISODES = 50
    
    # Class-level cache: {submission_id: all_metrics_dict}
    _metrics_cache: Dict[str, Dict] = {}
    
    def __init__(self, test_id: str, scenario_ids: List[str], benchmark_id: str):
        """Initialize with KPI-specific configuration."""
        super().__init__(test_id=test_id, benchmark_id=benchmark_id)
        self.scenario_ids = scenario_ids
        
        # Get KPI info from mapping
        self.kpi_info = ROBUSTNESS_RESILIENCE_KPI_MAPPING.get(test_id, {
            "name": "Unknown KPI",
            "metric_key": None,
            "description": ""
        })
        
        logger.info(
            f"Initialized RobustnessResilienceTestRunner\n"
            f"  Test ID: {test_id}\n"
            f"  KPI: {self.kpi_info['name']}"
        )
    
    def _compute_all_metrics(self, env, env_shift, agent) -> Dict:
        """
        Run complete multi-attacker evaluation and compute ALL metrics.

        Args:
            env: Grid2Op environment
            env_shift: Shift environment (ignored here)
            agent: Defender agent

        Returns:
            Dictionary containing ALL computed metrics for all 9 KPIs
        """
        logger.info(
            f"Starting complete evaluation in {self.__class__.__name__}\n"
            f"  Attackers: {self.ATTACKER_TYPES}\n"
            f"  Episodes: {self.NUM_EPISODES}"
        )

        # Import framework modules
        from evaluation_framework.result_getter import result_getter
        from evaluation_framework.metrics import metrics
        from attack_models.Environment import Environment

        # Wrap environment for attacker support
        wrapped_env = Environment(env, agent)

        # Load attackers
        attackers = self._load_attackers(wrapped_env, agent)

        # Run evaluation in temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Running episodes in: {temp_dir}")

            rg = result_getter(
                env=wrapped_env,
                defender=agent,
                n_episodes=self.NUM_EPISODES,
                save_folder=temp_dir,
                attackers=attackers
            )
            rg.calculate_metrics()

            # Load unperturbed baseline
            with open(os.path.join(temp_dir, "unperturbed.pkl"), "rb") as f:
                unperturbed_data = pickle.load(f)

            # Load and process metrics for each attacker
            metrics_list = []
            for attacker in attackers:
                pkl_path = os.path.join(temp_dir, attacker.pickle_file)
                with open(pkl_path, "rb") as f:
                    data_dict = pickle.load(f)

                m = metrics(
                    data_dict,
                    unperturbed_data,
                    wrapped_env.do_nothing_action(),
                    wrapped_env.get_similarity_score,
                    model_name=attacker.model_name
                )
                metrics_list.append(m)

            # Aggregate metrics across all attackers
            return self._aggregate_metrics(metrics_list, unperturbed_data)
    
    def _load_attackers(self, env, agent) -> List:
        """
        Load all attacker agents from the framework.
        
        Args:
            env: Environment wrapper instance
            agent: Defender agent
            
        Returns:
            List of attacker agent objects
        """
        logger.info(f"Loading {len(self.ATTACKER_TYPES)} attacker types")
        
        from attack_models.SACAttacker import SACAttacker
        from attack_models.PPOAttacker import PPOAttacker
        from attack_models.RLPerturbAttacker import RLPerturbAttacker
        from attack_models.GEPerturbAttacker import GEPerturbAttacker
        from attack_models.RPerturbAttacker import RPerturbAttacker
        from attack_models.LambdaPIRAttacker import LambdaPIRAttacker
        
        attackers = []
        trained_models_path = os.path.join(FRAMEWORK_PATH, "trained_models")
        
        # Attacker configurations
        attacker_configs = {
            "GEPerturb": lambda: GEPerturbAttacker(
                env=env.env, agent=agent, n_iter=10
            ),
            "LambdaPIR": lambda: LambdaPIRAttacker(
                model_path=os.path.join(trained_models_path, "SAC.zip"),
                env=env.env, agent=agent,
                lambda_param=0.7, initial_prob_policy=0.2, epsilon=1.0,
                gradient_step_size=0.1, refinement_iterations=20,
                decay_schedule="exponential", name="LambdaPIR", use_gpu=False
            ),
            "Random": lambda: RPerturbAttacker(
                env=env.env, prob_perturb=0.6
            ),
            "PPO": lambda: PPOAttacker(
                model_path=os.path.join(trained_models_path, "PPO.zip"),
                model_name="PPO", pickle_file="ppo.pkl"
            ),
            "SAC_10": lambda: SACAttacker(
                model_path=os.path.join(trained_models_path, "SAC.zip"),
                factor=10, model_name="SAC_10", pickle_file="sac_10.pkl"
            ),
            "SAC_5": lambda: SACAttacker(
                model_path=os.path.join(trained_models_path, "SAC.zip"),
                factor=5, model_name="SAC_5", pickle_file="sac_5.pkl"
            ),
            "RLPerturb": lambda: RLPerturbAttacker(
                model_path=os.path.join(trained_models_path, "RLPerturbAgent", "trained_rlpa_0.pth"),
                target_path=os.path.join(trained_models_path, "RLPerturbAgent", "trained_rlpa_target_net_0.pth"),
                env=env.env, agent=agent
            ),
        }
        
        for attacker_type in self.ATTACKER_TYPES:
            try:
                if attacker_type in attacker_configs:
                    attacker = attacker_configs[attacker_type]()
                    attackers.append(attacker)
                    logger.info(f"Loaded attacker: {attacker_type}")
            except Exception as e:
                logger.error(f"Failed to load attacker {attacker_type}: {e}")
        
        logger.info(f"Successfully loaded {len(attackers)} attackers")
        return attackers
    
    def _aggregate_metrics(self, metrics_list: List, unperturbed_data: Dict) -> Dict:
        """
        Aggregate metrics from all attackers into final results.
        
        Args:
            metrics_list: List of metrics objects from each attacker
            unperturbed_data: Dictionary with unperturbed episode data
            
        Returns:
            Dictionary with aggregated metrics for all 9 KPIs
        """
        logger.info(f"Aggregating metrics from {len(metrics_list)} attackers")
        
        # Collect metrics from each attacker
        vulnerability_scores = []
        steps_survived = []
        similarity_scores = []
        reward_drops = []
        action_change_freqs = []
        areas_between_curves = []
        degradation_times = []
        restoration_times = []
        state_similarities = []
        
        # Calculate total unperturbed reward once
        total_reward_unperturbed = sum([
            sum([r for r in ep if not np.isnan(r)]) 
            for ep in unperturbed_data['rewards']
        ])
        
        for m in metrics_list:
            logger.info(f"Processing metrics for attacker: {m.model_name}")
            
            # Robustness metrics
            vulnerability_scores.append(m.perturb_vulnerability.mean())
            steps_survived.append(m.metrics_robustness['n_steps'].mean())
            similarity_scores.append(m.metrics_robustness['similarity_score'].mean())
            
            # Reward drop calculation
            total_reward_perturbed = m.metrics_robustness['total_reward'].sum()
            if total_reward_unperturbed > 0:
                reward_drop = 100 * (total_reward_unperturbed - total_reward_perturbed) / total_reward_unperturbed
            else:
                reward_drop = 0.0
            reward_drops.append(reward_drop)
            
            # Action change frequency
            n_changed = m.metrics_robustness['n_actions_changed'].sum()
            n_total = m.metrics_robustness['n_steps_with_act'].sum()
            action_change_freqs.append(n_changed / n_total if n_total > 0 else 0)
            
            # Resilience metrics
            if 'area_per_1000_steps' in m.metrics_resilience.columns:
                areas_between_curves.append(m.metrics_resilience['area_per_1000_steps'].values[0])
            elif 'area' in m.metrics_resilience.columns:
                areas_between_curves.append(m.metrics_resilience['area'].values[0])
            else:
                areas_between_curves.append(0.0)
            
            degradation_times.append(m.metrics_resilience['degradation_time'].values[0])
            restoration_times.append(m.metrics_resilience['restoration_time'].values[0])
            state_similarities.append(np.mean([np.mean(ep) for ep in m.cos_similarity_all]))
        
        # Compute averages across all attackers
        aggregated = {
            # Robustness KPIs
            'perturb_vulnerability': np.mean(vulnerability_scores),
            'n_steps_survived': np.mean(steps_survived),
            'severity_of_change': 1.0 - np.mean(similarity_scores),  # Inverted (higher=worse)
            'reward_drop_percent': np.mean(reward_drops),
            'action_change_freq': np.mean(action_change_freqs),
            # Resilience KPIs
            'area_between_curves': np.mean(areas_between_curves),
            'degradation_time': np.mean(degradation_times),
            'restoration_time': np.mean(restoration_times),
            'state_similarity': np.mean(state_similarities),
        }
        
        logger.info(f"Aggregated metrics: {aggregated}")
        
        return aggregated


# ============================================================================
# Robustness KPI Classes (069-073)
# ============================================================================

class TestRunner_KPI_DF_069_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-DF-069: Drop-off in reward"""
    pass


class TestRunner_KPI_FF_070_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-FF-070: Frequency changed output AI agent"""
    pass


class TestRunner_KPI_SF_071_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-SF-071: Severity of changed output AI agent"""
    pass


class TestRunner_KPI_SF_072_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-SF-072: Steps survived with perturbations"""
    pass


class TestRunner_KPI_VF_073_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-VF-073: Vulnerability to perturbation"""
    pass


# ============================================================================
# Resilience KPI Classes (074-077)
# ============================================================================

class TestRunner_KPI_AF_074_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-AF-074: Area between reward curves"""
    pass


class TestRunner_KPI_DF_075_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-DF-075: Degradation time"""
    pass


class TestRunner_KPI_RF_076_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-RF-076: Restorative time"""
    pass


class TestRunner_KPI_SF_077_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-SF-077: Similarity state to unperturbed situation"""
    pass
