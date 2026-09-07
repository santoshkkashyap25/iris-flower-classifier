import os

class Config:
    """Base configuration class."""
    DEBUG = False
    TESTING = False
    SECRET_KEY = 'iris-flower-app'
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'iris.pkl')
    LOG_FILE = os.path.join(os.path.dirname(__file__), 'app.log')

    IRIS_SPECIES_MAP = {
        0: {
            "name": "Iris-setosa",
            "displayName": "Iris Setosa",
            "description": "Known for small petals and sepals. Native to Japan and Arctic regions. Often bright blue-purple or white.",
            "badgeColor": "#2563eb"
        },
        1: {
            "name": "Iris-versicolor",
            "displayName": "Iris Versicolor",
            "description": "Often found in North American wetlands. Features intermediate petal and sepal sizes with rich violet hues.",
            "badgeColor": "#059669"
        },
        2: {
            "name": "Iris-virginica",
            "displayName": "Iris Virginica",
            "description": "A robust species native to eastern North America with the largest petals and sepals among the three.",
            "badgeColor": "#7c3aed"
        },
    }

    IRIS_FEATURE_NAMES = [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ]

    IRIS_FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # Legacy mapping for backwards compatibility with 'a', 'b', 'c', 'd'
    LEGACY_KEY_MAP = {
        'a': 'sepal_length',
        'b': 'sepal_width',
        'c': 'petal_length',
        'd': 'petal_width'
    }

    IRIS_FEATURE_RANGES = {
        'sepal length (cm)': {'key': 'sepal_length', 'label': 'Sepal Length', 'min': 4.3, 'max': 7.9, 'step': 0.1, 'default': 5.8},
        'sepal width (cm)': {'key': 'sepal_width', 'label': 'Sepal Width', 'min': 2.0, 'max': 4.4, 'step': 0.1, 'default': 3.0},
        'petal length (cm)': {'key': 'petal_length', 'label': 'Petal Length', 'min': 1.0, 'max': 6.9, 'step': 0.1, 'default': 4.3},
        'petal width (cm)': {'key': 'petal_width', 'label': 'Petal Width', 'min': 0.1, 'max': 2.5, 'step': 0.1, 'default': 1.3}
    }

    IRIS_PRESETS = [
        {
            "id": "setosa",
            "name": "Setosa Sample",
            "species": "Iris-setosa",
            "features": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        },
        {
            "id": "versicolor",
            "name": "Versicolor Sample",
            "species": "Iris-versicolor",
            "features": {"sepal_length": 5.9, "sepal_width": 2.8, "petal_length": 4.2, "petal_width": 1.3}
        },
        {
            "id": "virginica",
            "name": "Virginica Sample",
            "species": "Iris-virginica",
            "features": {"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.5, "petal_width": 2.0}
        }
    ]

