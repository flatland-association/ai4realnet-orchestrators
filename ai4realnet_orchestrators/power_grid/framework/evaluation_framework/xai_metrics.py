import gc
from typing import Dict,  Callable
import numpy as np
import pandas as pd
import quantus
import tensorflow as tf
from ai4realnet_orchestrators.power_grid.framework.evaluation_framework.xai_feature_attribution import tf_explain_func
import logging

logger = logging.getLogger(__name__)


class XAIEvaluatorTF:
    """
    Evaluator class for XAI metrics tailored for TensorFlow models.
    """

    def __init__(
            self,
            model: tf.keras.Model,
            features: np.ndarray,
            targets: np.ndarray,
            attributions: Dict[str, np.ndarray],
            num_classes: int,
            softmax: bool = False,
            max_workers: int = 1,
    ):

        self.model = build_fast_predict_model(build_action_only_model(model))
        self.features = features
        self.targets = targets
        self.attributions = attributions
        self.num_classes = num_classes
        self.softmax = softmax
        self.max_workers = max_workers

        self.evaluation_metrics: Dict[str, Callable] = None
        self.results: pd.DataFrame = None
        self.std_results: pd.DataFrame = None
        self.xai_methods: Dict[str, np.ndarray] = None

        self.initialize_xai_methods()
        self.setup_evaluation_metrics()

        self.device = "cpu"

    def initialize_xai_methods(self):
        """
        Initializes the XAI methods by referencing the pre-computed attribution arrays.
        """
        self.xai_methods = {
            "InputXGradient": self.attributions.get("InputXGradient"),
            "Saliency": self.attributions.get("Saliency"),
        }

    def setup_evaluation_metrics(self):
        """
        Sets up the evaluation metrics from Quantus library to be used in the evaluation process.
        """
        self.evaluation_metrics = {
            "Average\nSensitivity (Ro)": quantus.AvgSensitivity(),
            "Faithfulness\nEstimate (F)": quantus.FaithfulnessEstimate(perturb_baseline="mean")
        }

    def compute_scores(self, method_name, a_batch, metric_name, metric_func):
        """
        Computes the scores for a given XAI method and metric using pre-computed attributions.
        """
        gc.collect()

        scores = metric_func(
            model=self.model,
            x_batch=self.features,
            y_batch=self.targets,
            a_batch=a_batch,
            softmax=self.softmax,
            device=self.device,
            explain_func=tf_explain_func,
            explain_func_kwargs={
                "method": method_name,
                "abs": method_name == 'Saliency'
            },
        )
        return method_name, metric_name, scores

    def evaluate(self):
        """
        Evaluates the XAI methods using the defined metrics.
        """
        results = {method_name: {} for method_name in self.xai_methods}

        for method_name, a_batch in self.xai_methods.items():
            if a_batch is None:
                continue
            for metric_name, metric_func in self.evaluation_metrics.items():
                clean_name = metric_name.replace("\n", " ")
                logger.info(f"Evaluating {clean_name} with {method_name}...")
                method_name_res, metric_name_res, scores = self.compute_scores(
                    method_name, a_batch, metric_name, metric_func
                )
                results[method_name_res][metric_name_res] = scores

        # aggregation
        results_agg = {}
        std_results_agg = {}

        for method in results:
            results_agg[method] = {
                metric: np.nanmean([value for value in values if np.isfinite(value)])
                for metric, values in results[method].items()
            }
            std_results_agg[method] = {
                metric: np.nanstd([value for value in values if np.isfinite(value)])
                for metric, values in results[method].items()
            }

        self.results = pd.DataFrame.from_dict(results_agg).T
        self.std_results = pd.DataFrame.from_dict(std_results_agg).T


def build_action_only_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Reduces Multi-Output-Model to action_out (logits) as Functional-Model.
    """
    if model.name == "action_only":
        return model
    inputs = model.inputs
    out = model(inputs)
    action = out[0] if isinstance(out, (list, tuple)) else out
    action = tf.keras.layers.Activation("linear", name="action_activation")(action)
    return tf.keras.Model(inputs, action, name="action_only")


def build_fast_predict_model(inner: tf.keras.Model) -> tf.keras.Model:
    """
    Compiled predict path; `linear_top` remains the outermost layer.
    """
    if inner.name == "fast_predict":
        return inner

    inp = tf.keras.Input(shape=inner.input_shape[1:], dtype=tf.float32)
    out = inner(inp)
    out = tf.keras.layers.Activation("linear", name="linear_top")(out)
    fast = tf.keras.Model(inp, out, name="fast_predict")

    call_fn = tf.function(
        lambda t: fast(t, training=False),
        input_signature=[tf.TensorSpec([None, *inner.input_shape[1:]], tf.float32)],
    )

    def fast_predict(x, **kwargs):
        return call_fn(tf.convert_to_tensor(x, dtype=tf.float32)).numpy()

    fast.predict = fast_predict
    return fast

