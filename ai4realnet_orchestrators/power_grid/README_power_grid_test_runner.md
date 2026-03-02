# Power Grid Test Runner

Combined KPI implementations for the AI4REALNET Power Grid domain.

## Overview

This module provides the base `PowerGridTestRunner` class and implementations for 12 KPIs across three categories:

| Category | KPIs | Method |
|----------|------|--------|
| Operational | 008, 012, 036 | ScoreL2RPN2023 |
| Robustness | 069-073 | Multi-attacker framework |
| Resilience | 074-077 | Multi-attacker framework |

## KPIs Implemented

### Operational KPIs
| Class | KPI | Description |
|-------|-----|-------------|
| `TestRunner_KPI_AF_008_Power_Grid` | Assistant alert accuracy | Alert confidence score |
| `TestRunner_KPI_CF_012_Power_Grid` | Carbon intensity | Non-renewable energy score |
| `TestRunner_KPI_OF_036_Power_Grid` | Operation score | Grid operation performance |

### Robustness KPIs (Benchmark: 3810191b-8cfd-4b03-86b2-f7e530aab30d)
| Class | KPI | Description |
|-------|-----|-------------|
| `TestRunner_KPI_DF_069_Power_Grid` | Drop-off in reward | % decrease in reward under attack |
| `TestRunner_KPI_FF_070_Power_Grid` | Frequency changed output | Action change frequency [0-1] |
| `TestRunner_KPI_SF_071_Power_Grid` | Severity of changed output | Severity of action changes [0-1] |
| `TestRunner_KPI_SF_072_Power_Grid` | Steps survived | Timesteps before failure |
| `TestRunner_KPI_VF_073_Power_Grid` | Vulnerability to perturbation | Proportion vulnerable [0-1] |

### Resilience KPIs (Benchmark: 31ea606b-681a-437a-85b9-7c81d4ccc287)
| Class | KPI | Description |
|-------|-----|-------------|
| `TestRunner_KPI_AF_074_Power_Grid` | Area between curves | Integrated performance degradation |
| `TestRunner_KPI_DF_075_Power_Grid` | Degradation time | Time until performance degrades |
| `TestRunner_KPI_RF_076_Power_Grid` | Restorative time | Time to restore performance |
| `TestRunner_KPI_SF_077_Power_Grid` | State similarity | Cosine similarity to unperturbed [-1,1] |

## Architecture

```
PowerGridTestRunner (Base Template)
├── init() ─────────────── Loads JSON submission metadata
├── run_scenario() ─────── Creates env + loads agent
├── load_agent() ────────── Extracts zip, loads model
└── getResult() [abstract]

RobustnessResilienceTestRunner (Extended)
└── getResult(env, agent)
    ├── Loads 7 attackers
    ├── Runs 50 episodes each
    ├── Aggregates metrics
    └── Returns KPI-specific value
```

## Attackers (Robustness/Resilience)

| Attacker | Type | Description |
|----------|------|-------------|
| GEPerturb | White-box | Gradient estimation perturbation |
| LambdaPIR | Hybrid | Lambda policy iteration with refinement |
| Random | Baseline | Random observation perturbations |
| PPO | RL-based | PPO-trained adversarial agent |
| SAC_5 | RL-based | SAC-trained (factor=5) |
| SAC_10 | RL-based | SAC-trained (factor=10) |
| RLPerturb | RL-based | Deep Q-learning perturbation agent |

## Configuration

### Robustness/Resilience Settings
```python
ATTACKER_TYPES = ["GEPerturb", "LambdaPIR", "Random", "PPO", "SAC_10", "SAC_5", "RLPerturb"]
NUM_EPISODES = 50
```

### Required Files
```
power_grid/
├── framework/
│   ├── trained_models/      # Attacker models (SAC.zip, PPO.zip, etc.)
│   ├── attack_models/       # Attacker implementations
│   ├── evaluation_framework/
│   └── modified_curriculum_classes/
├── configuration/
│   └── scoring-config.json  # For operational KPIs
└── power_grid_test_runner.py
```

## Usage

### Via FAB Orchestrator (Production)
```python
# In __init__.py
from .power_grid_test_runner import TestRunner_KPI_VF_073_Power_Grid

power_grid_orchestrator = Orchestrator(
    test_runners={
        "b8a9a411-7cfe-4c1d-b9a6-eef1c0efe920": TestRunner_KPI_VF_073_Power_Grid(
            test_id="b8a9a411-7cfe-4c1d-b9a6-eef1c0efe920",
            scenario_ids=['61063867-df62-4024-be42-c57507a15d7c'],
            benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        ),
    }
)
```

### Local Testing
See `test_local_robustness_resilience_kpi_069_077.py` for standalone testing without the orchestrator.

## Authors

- **Robustness/Resilience KPIs (069-077)**: INESC TEC
- **Operational KPIs (008, 012, 036)**: AI4REALNET Consortium
