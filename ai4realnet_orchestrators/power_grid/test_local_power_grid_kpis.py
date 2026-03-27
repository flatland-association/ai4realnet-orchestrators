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
    evaluate_operational_kpis
)

robustness_runner = RobustnessResilienceTestRunner(
    test_id="b8a9a411-7cfe-4c1d-b9a6-eef1c0efe920",
    scenario_ids=['61063867-df62-4024-be42-c57507a15d7c'],
    benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
)

robustness_runner.init(
    submission_data_url="https://raw.githubusercontent.com/flatland-association/ai4realnet-orchestrators/refs/heads/merged-powergrid-kpis/ai4realnet_orchestrators/power_grid/configuration/curriculum-ai4realnet-small.json",
    submission_id="local-test"
)

# OVERRIDE framework path for local testing
robustness_runner.FRAMEWORK_PATH = FRAMEWORK_DIR

print(f"✅ TestRunner initialized")
print(f"   KPI: {robustness_runner.kpi_info['name']}")
print(f"   Framework: {robustness_runner.FRAMEWORK_PATH}")

env_path = "/mnt/d/PythonProjects/ai4realnet/grid2op-scenario/ai4realnet_small"
print(f"\n📦 Environment path: {env_path}")
print(f"   Exists: {os.path.exists(env_path)}")

agent_path = os.path.join(SCRIPT_DIR, "submission", "trained_model", "curriculum", "agent.zip")
print(f"\n📦 Agent path: {agent_path}")
print(f"   Exists: {os.path.exists(agent_path)}")

try:
    print(f"\n🔄 Loading agent from zip (local)...")
    
    # Extract zip locally
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(agent_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    print(f"   Extracted to: {temp_dir}")
    print(f"   Contents: {os.listdir(temp_dir)}")
    
    # Load the agent using the framework's method
    from modified_curriculum_classes.baseline import CurriculumAgent
    import grid2op
    from lightsim2grid import LightSimBackend
    
    # Create environment
    if os.path.exists(env_path):
        env = grid2op.make(env_path, backend=LightSimBackend())
    else:
        env = grid2op.make(robustness_runner.ENV_NAME)
    
    # Create and load agent
    agent = CurriculumAgent(env.action_space, env.observation_space, "test_agent")
    
    # Find model and actions paths in extracted zip
    model_path = os.path.join(temp_dir, "model")
    actions_path = os.path.join(temp_dir, "actions")
    
    agent.load(model_path, actions_path, best_action_threshold=0.95)
    
    robustness_runner._defender_agent = agent
    
    print("✅ Agent loaded!")

    print("\n🔄 Running operational evaluation (this may take a while)...")
    operational_result = evaluate_operational_kpis(env, agent)

    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"Result: {operational_result}")

    print("\n🔄 Running robustness evaluation (this may take a while)...")
    robustness_result = robustness_runner.run_scenario(
        scenario_id="61063867-df62-4024-be42-c57507a15d7c",
        submission_id="local-test-001"
    )
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"Result: {robustness_result}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()