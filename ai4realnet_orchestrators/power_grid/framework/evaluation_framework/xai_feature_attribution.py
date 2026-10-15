from typing import Dict, Optional
import numpy as np
import tensorflow as tf


class FeatureAttributionTF:
    """
    A class for computing feature attributions using various algorithms (TensorFlow version).

    Parameters:
        model (tf.keras.Model): Single-output Keras model (e.g. from build_action_only_model).
        test_features (np.ndarray): Test data for which attributions will be calculated.
        test_targets (np.ndarray): Targets for the test data.
    """

    def __init__(
            self,
            model: tf.keras.Model,
            test_features: np.ndarray,
            test_targets: np.ndarray,
    ) -> None:
        self.test_features = test_features
        self.test_targets = test_targets
        self.model = model
        self.attributions: Dict[str, np.ndarray] = {}

    def _gradients(self) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Compute gradients of the target logits w.r.t. The input features.
        """
        features = tf.convert_to_tensor(self.test_features, dtype=tf.float32)
        targets = tf.convert_to_tensor(self.test_targets, dtype=tf.int32)

        with tf.GradientTape() as tape:
            tape.watch(features)
            logits = self.model(features, training=False)
            indices = tf.stack([tf.range(tf.shape(logits)[0]), targets], axis=1)
            outputs = tf.gather_nd(logits, indices)

        return features, tape.gradient(outputs, features)

    def input_x_gradient(self) -> np.ndarray:
        """
        Compute feature attributions using the Input x Gradients algorithm.

        Returns:
            np.ndarray: Feature attributions for the test data.
        """
        features, gradients = self._gradients()
        return (features * gradients).numpy()

    def saliency(self) -> np.ndarray:
        """
        Compute feature attributions using the Saliency algorithm.

        Returns:
            np.ndarray: Feature attributions for the test data.
        """
        _, gradients = self._gradients()
        return np.abs(gradients.numpy())

    def calculate_all_attributions(
            self, feature_mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute and store feature attributions using all defined methods.

        Returns:
            dict: A dictionary containing all attributions for the test data.
        """
        self.attributions["InputXGradient"] = self.input_x_gradient()
        self.attributions["Saliency"] = self.saliency()

        return self.attributions


def tf_explain_func(model, inputs, targets, **kwargs):
    """
    Adapter: uses FeatureAttributionTF as Quantus-explain_func.
    'model' from Quantus (= ActionOnlyModel, Single-Output).
    """
    method = kwargs.get("method", "Saliency")
    fa = FeatureAttributionTF(
        model=model,
        test_features=np.asarray(inputs, dtype=np.float32),
        test_targets=np.asarray(targets).astype(np.int32),
    )
    if method == "InputXGradient":
        a = fa.input_x_gradient()
    elif method == "Saliency":
        a = fa.saliency()
    else:
        raise ValueError(f"Unsupported method: {method}")
    if kwargs.get("abs", False):
        a = np.abs(a)
    return a
