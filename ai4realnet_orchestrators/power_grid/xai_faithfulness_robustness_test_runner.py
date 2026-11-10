import logging
from typing import Dict
import numpy as np
import pandas as pd
import tensorflow as tf

from ai4realnet_orchestrators.power_grid.power_grid_test_runner import PowerGridTestRunner
from ai4realnet_orchestrators.power_grid.framework.evaluation_framework.xai_feature_attribution import FeatureAttributionTF
from ai4realnet_orchestrators.power_grid.framework.evaluation_framework.xai_metrics import build_action_only_model, XAIEvaluatorTF

logger = logging.getLogger(__name__)


# KPI ID to metric mapping
XAI_ROBUSTNESS_FAITHFULNESS_KPI_MAPPING = {
    "89919375-8b53-4e3f-8382-a97e0af7eb56": {
        "name": "KPI-EF-086: Explainability Robustness",
        "metric_key": "explainability_robustness",
        "description": "Stability of feature attributions under small input perturbations [0-1, lower=better]"
    },
    "ce7d0394-1aa0-41c2-84b7-dfcfe006eb8b": {
        "name": "KPI-EF-087: Explainability Faithfulness",
        "metric_key": "explainability_faithfulness",
        "description": "Agreement between attributed feature importance and the agent's actual output change when "
                       "those features are removed [0-1, higher=better]"
    }
}


def vect_colnames(obs):
    """
    Build column names matching the layout of ``obs.to_vect()``.

    Args:
        obs: Grid2Op observation used as template for the vector layout.

    Returns:
        list[str]: Feature names, in the same order as ``obs.to_vect()``.
    """
    names = []
    for a in obs.attr_list_vect:
        size = np.asarray(getattr(obs, a)).ravel().size
        if size == 1:
            names.append(a)
        else:
            names.extend(f"{a}_{i}" for i in range(size))
    return names


def build_dataset(env, agent, expected_dim=None, dataset_size=10_000, rho_threshold=0.9, seed=42):
    """
   Collect a dataset of near-critical grid states by running the agent in the environment.

   Args:
       env: Grid2Op environment to run.
       agent: Agent providing ``act(obs, reward, done)``; seeded if it exposes ``seed``.
       Expected_dim (int, optional): Asserted number of features of the observation vector.
       Dataset_size (int): Maximum number of samples to collect.
       Rho_threshold (float): Minimum max-rho required for an observation to be kept.
       Seed (int): Seed for environment, NumPy and agent for reproducibility.

   Returns:
       pandas.DataFrame: Deduplicated float32 observation vectors with named columns.
       """
    env.seed(seed)
    np.random.seed(seed)
    if hasattr(agent, "seed"):
        agent.seed(seed)

    obs0 = env.reset()
    colnames = vect_colnames(obs0)

    if expected_dim is not None:
        assert len(colnames) == expected_dim, f"{len(colnames)} features != {expected_dim} expected"

    rows = []

    for chronic_id in range(len(env.chronics_handler.subpaths)):
        if len(rows) >= dataset_size:
            break

        env.set_id(chronic_id)
        obs = env.reset()
        done, reward, step = False, 0.0, 0

        while not done and len(rows) < dataset_size:
            action = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(action)
            step += 1

            if obs.rho.max() >= rho_threshold:
                rows.append(obs.to_vect().astype(np.float32))

    df = pd.DataFrame(np.asarray(rows, dtype=np.float32), columns=colnames)

    keep = ~df.duplicated()
    return df[keep].reset_index(drop=True)


class XAIRobustnessFaithfulnessTestRunner(PowerGridTestRunner):
    _metrics_cache: Dict[str, Dict] = {}
    # Specific KPI mapping for this category
    KPI_MAPPING = XAI_ROBUSTNESS_FAITHFULNESS_KPI_MAPPING

    def _compute_all_metrics(self, env, env_shift, agent) -> Dict:
        """
        Compute all metrics for explainability robustness and explainability faithfulness.

        Args:
            env: Grid2Op environment
            env_shift: Grid2Op shift environment (if applicable)
            agent: Loaded agent

        Returns:
            dict containing all computed metrics for both KPIs
        """

        # Create dataset
        dataset = build_dataset(env, agent)
        test_features_numpy = dataset.to_numpy().astype(np.float32)

        # Load TensorFlow Keras model
        keras_model = agent.agent.model
        keras_model.trainable = False
        action_model = build_action_only_model(keras_model)

        # Prepare a subset for evaluation
        rng = np.random.default_rng(42)
        n_eval = min(512, len(test_features_numpy))
        idx = rng.choice(len(test_features_numpy), size=n_eval, replace=False)
        idx.sort()
        features_eval = test_features_numpy[idx]

        # Forward pass
        test_logits_tf = action_model(features_eval, training=False)

        # Apply softmax to convert logits to probabilities for each class
        test_probabilities_tf = tf.nn.softmax(test_logits_tf, axis=1)

        # Use argmax to find the class with the highest probability for each sample
        test_targets_numpy = tf.argmax(test_probabilities_tf, axis=1).numpy()

        # Calculate Feature Attributions
        logger.info("Calculating Feature Attributions...")
        fa = FeatureAttributionTF(
            model=action_model,
            test_features=features_eval,
            test_targets=test_targets_numpy
        )
        attributions = fa.calculate_all_attributions()

        # Evaluation
        logger.info("Calculating explainability KPIs...")
        evaluator = XAIEvaluatorTF(
            model=action_model,
            features=features_eval,
            targets=test_targets_numpy,
            attributions=attributions,
            num_classes=int(test_logits_tf.shape[1]),
            softmax=False,
            max_workers=5
        )
        evaluator.evaluate()

        return {
            'explainability_robustness': evaluator.results['Average\nSensitivity (Ro)'].mean(),
            'explainability_faithfulness': evaluator.results['Faithfulness\nEstimate (F)'].mean()
        }


class TestRunner_KPI_EF_086_Power_Grid(XAIRobustnessFaithfulnessTestRunner):
    """KPI-EF-086: Explainability Robustness"""
    pass


class TestRunner_KPI_EF_087_Power_Grid(XAIRobustnessFaithfulnessTestRunner):
    """KPI-EF-087: Explainability Faithfulness"""
    pass
