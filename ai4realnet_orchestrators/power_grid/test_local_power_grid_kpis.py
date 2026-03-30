"""
Local Test Script for Robustness & Resilience Framework
========================================================

Runs the multi-attacker evaluation framework LOCALLY without the FAB 
orchestrator. Use for development, debugging, and testing agents before 
submission.

Usage:
    python test_local.py

Note: 
    This bypasses Celery/RabbitMQ orchestration. For production, use FAB.

Author: INESC TEC
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "0"

import zipfile
import tempfile

# ============================================================
# PATH SETUP - Must be done BEFORE any imports
# ============================================================

# Get the directory where this script lives (power_grid folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.join(SCRIPT_DIR, "framework")
ORCHESTRATORS_DIR = os.path.dirname(SCRIPT_DIR)  # ai4realnet_orchestrators
PARENT_DIR = os.path.dirname(ORCHESTRATORS_DIR)  # ai4realnet-orchestrators

# Add paths for imports
sys.path.insert(0, PARENT_DIR)          # For ai4realnet_orchestrators package
sys.path.insert(0, ORCHESTRATORS_DIR)   # For base test_runner
sys.path.insert(0, FRAMEWORK_DIR)       # For attack_models, etc.

# Change working directory to framework so relative paths work
os.chdir(FRAMEWORK_DIR)

print("=" * 60)
print("LOCAL TEST - FAB TestRunner")
print("=" * 60)
print(f"Working directory: {os.getcwd()}")
print(f"Framework directory: {FRAMEWORK_DIR}")

# ============================================================
# NOW we can import (paths are set up)
# ============================================================

from ai4realnet_orchestrators.power_grid.power_grid_test_runner import (
    RobustnessResilienceTestRunner,
    evaluate_operational_kpis,
    evaluate_domain_shift_kpis,
    ReliabilityTestRunner,
)

from ExpertAgent.utils.helper_functions import make_gymenv
from ExpertAgent.ExpertAgent import ExpertAgentRL
from stable_baselines3.ppo import MlpPolicy

reliability_runner = ReliabilityTestRunner(test_id="855729a4-6729-4ae2-bb8d-443ef4867d94", 
                                           scenario_ids=['81f18394-0164-4896-9408-4315bcfcc5e0'], 
                                           benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c")

reliability_runner.init(
    submission_data_url="https://raw.githubusercontent.com/flatland-association/ai4realnet-orchestrators/refs/heads/milad-merged-powergrid-kpis/ai4realnet_orchestrators/power_grid/configuration/expert-ai4realnet-small.json",
    submission_id="local_test_reliability"
)


print(f"✅ TestRunner initialized")

env_path = "/home/milad/data_grid2op/ai4realnet_small"
print(f"\n📦 Environment path: {env_path}")
print(f"   Exists: {os.path.exists(env_path)}")

agent_path = os.path.join(SCRIPT_DIR, "submission", "trained_model", "ExpertAgent", "ai4realnet_small", "PPO_SB3.zip")
print(f"\n📦 Agent path: {agent_path}")
print(f"   Exists: {os.path.exists(agent_path)}")

try:
    print(f"\n🔄 Loading agent from zip (local)...")
    import grid2op
    from lightsim2grid import LightSimBackend
    # Create environment
    if os.path.exists(env_path):
        env = grid2op.make(env_path, backend=LightSimBackend())
        env_shift = grid2op.make(env_path, backend=LightSimBackend())
    
    # create and load the agent
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
    
    print("✅ Agent loaded!")

    print("\n🔄 Running Reilability evaluation (this may take a while)...")
    reliability_result = reliability_runner.run_scenario(
        scenario_id="81f18394-0164-4896-9408-4315bcfcc5e0",
        submission_id="local-test-adaptation-time"
    )
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"Adaptation time: {reliability_result}")
    
    reliability_result = reliability_runner.run_scenario(
        scenario_id="4d2b00cd-447a-4c7e-8cab-863f0402cb67",
        submission_id="local-test-performance-drop"
    )
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"Performance drop: {reliability_result}")
    
    # print("\n🔄 Running reliability evaluation (this may take a while)...")
    # reliability_result = evaluate_domain_shift_kpis(env, env_shift, agent)

    # print("\n" + "=" * 60)
    # print("✅ EVALUATION COMPLETE!")
    # print("=" * 60)
    # print(f"Result: {reliability_result}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    