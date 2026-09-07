import os
import pickle
import threading
import logging
import numpy as np
import pandas as pd
from config import Config

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()

def load_iris_model(model_path=None):
    """Thread-safe loader for the trained Iris classification model."""
    global _model
    if model_path is None:
        model_path = Config.MODEL_PATH

    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info(f"Loading model from: {model_path}")
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found at {model_path}. Please run train_model.py first.")
                    raise FileNotFoundError(f"Model file not found at {model_path}")
                try:
                    with open(model_path, 'rb') as file:
                        _model = pickle.load(file)
                    logger.info("Iris model pipeline loaded successfully.")
                except Exception as e:
                    logger.error(f"Error loading model from {model_path}: {e}")
                    raise RuntimeError(f"Failed to load model: {e}")
    return _model

def get_feature_names():
    """Returns the ordered list of feature names expected by the model."""
    return Config.IRIS_FEATURE_NAMES

def predict_iris(features):
    """Makes a species prediction from the provided 4 numerical measurements.

    Args:
        features: List or array-like of 4 numeric values [sepal_length, sepal_width, petal_length, petal_width].

    Returns:
        tuple: (predicted_class_index, probabilities_list) or (-1, []) on failure.
    """
    global _model
    if _model is None:
        try:
            load_iris_model(Config.MODEL_PATH)
        except Exception as e:
            logger.error(f"Prediction aborted: Unable to load model: {e}")
            return -1, []

    try:
        if len(features) != 4:
            logger.error(f"Invalid input shape: expected 4 features, got {len(features)}")
            return -1, []

        float_features = [float(v) for v in features]
        input_df = pd.DataFrame([float_features], columns=Config.IRIS_FEATURE_NAMES)
        prediction_proba = _model.predict_proba(input_df)[0]
        predicted_class = int(np.argmax(prediction_proba))
        proba_list = [float(p) for p in prediction_proba]

        logger.info(
            f"Prediction succeeded: Features={features} -> Class={predicted_class}, "
            f"Probabilities={proba_list}"
        )
        return predicted_class, proba_list
    except (ValueError, TypeError) as e:
        logger.error(f"Validation error during prediction with features {features}: {e}")
        return -1, []
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        return -1, []
