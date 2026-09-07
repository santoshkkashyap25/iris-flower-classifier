import os
import logging
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from config import Config
from model import load_iris_model, predict_iris, get_feature_names

def setup_logging(app):
    """Configures centralized application logging without duplicate handlers."""
    if logging.getLogger().hasHandlers():
        return

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    
    # Optionally write to log file if configured and not in test mode
    log_file = app.config.get('LOG_FILE')
    if log_file and not app.config.get('TESTING'):
        try:
            file_handler = logging.FileHandler(log_file)
            handlers.append(file_handler)
        except Exception:
            pass

    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers)

def get_confidence_message(prob):
    """Generates an intuitive confidence summary based on the highest probability."""
    if prob >= 0.95:
        return "Very High Confidence — The measurements align strongly with this species profile."
    elif prob >= 0.80:
        return "High Confidence — Characteristics match this species well."
    elif prob >= 0.60:
        return "Moderate Confidence — The specimen exhibits slight boundary overlap."
    else:
        return "Low Confidence — Measurements lie near the classification decision boundary."

def extract_features_from_dict(data):
    """Extracts the 4 required flower dimensions from a dictionary supporting

    both semantic names ('sepal_length', etc.) and legacy aliases ('a', 'b', 'c', 'd').
    """
    feature_keys = Config.IRIS_FEATURE_KEYS
    legacy_map = Config.LEGACY_KEY_MAP

    values = []
    for key in feature_keys:
        # Check semantic key first, then fallback to legacy key (e.g. 'a')
        if key in data and data[key] not in (None, ''):
            values.append(float(data[key]))
        else:
            # Find matching legacy key
            legacy_key = next((k for k, v in legacy_map.items() if v == key), None)
            if legacy_key and legacy_key in data and data[legacy_key] not in (None, ''):
                values.append(float(data[legacy_key]))
            else:
                raise KeyError(f"Missing required measurement: '{key}'")
    return values

def create_app(config_class=Config):
    """Flask application factory for personal project."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    setup_logging(app)
    logger = logging.getLogger(__name__)

    # Pre-load model on application initialization
    with app.app_context():
        try:
            load_iris_model(app.config.get('MODEL_PATH'))
            logger.info("Model pipeline pre-loaded successfully during app startup.")
        except Exception as e:
            logger.warning(f"Could not pre-load model at startup (will retry on first request): {e}")

    @app.route('/')
    def home():
        """Renders the main interactive prediction page."""
        return render_template(
            'home.html',
            feature_ranges=app.config['IRIS_FEATURE_RANGES'],
            presets=app.config.get('IRIS_PRESETS', []),
            species_map=app.config['IRIS_SPECIES_MAP']
        )

    @app.route('/predict', methods=['GET', 'POST'])
    def predict():
        """Handles web form submission for prediction."""
        if request.method == 'GET':
            return redirect(url_for('home'))

        try:
            features = extract_features_from_dict(request.form)
            
            # Check for range anomalies to notify user with a friendly flash warning
            feature_ranges = app.config['IRIS_FEATURE_RANGES']
            for name, val in zip(Config.IRIS_FEATURE_NAMES, features):
                range_info = feature_ranges.get(name)
                if range_info and not (range_info['min'] <= val <= range_info['max']):
                    flash(
                        f"Notice: {range_info['label']} ({val} cm) is outside typical range "
                        f"({range_info['min']} – {range_info['max']} cm). Prediction may be less certain.",
                        'warning'
                    )

            predicted_class, probabilities = predict_iris(features)

            if predicted_class == -1:
                flash("Prediction failed: Could not load the model. Please ensure the model is trained.", 'error')
                return redirect(url_for('home'))

            species_info = app.config['IRIS_SPECIES_MAP'].get(predicted_class, {
                "name": "Unknown",
                "displayName": "Unknown Species",
                "description": "No species description available."
            })

            max_prob = probabilities[predicted_class]
            confidence_message = get_confidence_message(max_prob)

            probabilities_display = []
            for idx, prob in enumerate(probabilities):
                info = app.config['IRIS_SPECIES_MAP'].get(idx, {})
                probabilities_display.append({
                    "name": info.get("displayName", f"Species {idx}"),
                    "raw_species": info.get("name", ""),
                    "probability_pct": f"{prob * 100:.1f}%",
                    "probability_val": round(prob * 100, 1),
                    "badge_color": info.get("badgeColor", "#6b7280")
                })

            # Sort probabilities descending for clean display
            probabilities_display.sort(key=lambda x: x["probability_val"], reverse=True)

            measurements = [
                {"label": "Sepal Length", "val": features[0]},
                {"label": "Sepal Width", "val": features[1]},
                {"label": "Petal Length", "val": features[2]},
                {"label": "Petal Width", "val": features[3]},
            ]

            return render_template(
                'after.html',
                species_info=species_info,
                probabilities=probabilities_display,
                confidence_message=confidence_message,
                confidence_pct=f"{max_prob * 100:.1f}%",
                measurements=measurements
            )

        except (ValueError, KeyError) as e:
            flash(f"Invalid input: {e}. Please enter valid numerical values.", 'error')
            return redirect(url_for('home'))
        except Exception as e:
            logger.exception(f"Unexpected prediction error: {e}")
            flash("An unexpected error occurred during prediction. Please try again.", 'error')
            return redirect(url_for('home'))

    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        """REST API endpoint returning JSON predictions."""
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Request payload must be JSON with 'Content-Type: application/json'"
            }), 400

        data = request.get_json() or {}

        try:
            if "features" in data and isinstance(data["features"], list):
                if len(data["features"]) != 4:
                    return jsonify({
                        "success": False,
                        "error": f"Expected 4 numeric values in 'features', got {len(data['features'])}"
                    }), 400
                features = [float(v) for v in data["features"]]
            else:
                features = extract_features_from_dict(data)

            predicted_class, probabilities = predict_iris(features)

            if predicted_class == -1:
                return jsonify({
                    "success": False,
                    "error": "Model prediction failed. Verify model file exists and is valid."
                }), 500

            species_info = app.config['IRIS_SPECIES_MAP'].get(predicted_class, {})
            max_prob = probabilities[predicted_class]

            return jsonify({
                "success": True,
                "prediction": {
                    "class_id": predicted_class,
                    "species": species_info.get("name"),
                    "display_name": species_info.get("displayName"),
                    "confidence": round(max_prob, 4),
                    "confidence_percentage": f"{max_prob * 100:.1f}%",
                    "description": species_info.get("description")
                },
                "probabilities": {
                    app.config['IRIS_SPECIES_MAP'][i]['name']: round(prob, 4)
                    for i, prob in enumerate(probabilities)
                },
                "input_features": {
                    "sepal_length": features[0],
                    "sepal_width": features[1],
                    "petal_length": features[2],
                    "petal_width": features[3]
                }
            }), 200

        except (ValueError, KeyError) as e:
            return jsonify({"success": False, "error": str(e)}), 400
        except Exception as e:
            logger.exception(f"API prediction error: {e}")
            return jsonify({"success": False, "error": "Internal server error"}), 500

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404

    return app

# Standard application instance for Gunicorn and direct running
app = create_app()

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)