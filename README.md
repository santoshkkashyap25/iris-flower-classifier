# Iris Flower Species Predictor

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-00c7b7?style=for-the-badge&logo=render&logoColor=white)](https://flotect.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

🔗 **Live Application:** [https://flotect.onrender.com/](https://flotect.onrender.com/)

An interactive machine learning web application built with **Flask** and **Scikit-Learn** that predicts the species of an Iris flower based on its sepal and petal measurements.

The application features a modern botanical user interface, real-time input sliders with quick-archetype presets, a trained Scikit-Learn classification pipeline (`StandardScaler` + `LogisticRegression`), a dedicated JSON REST API, and an automated test suite.

---

## Features

- **Interactive Sliders**: User-friendly range sliders for entering Sepal and Petal dimensions (Length and Width in centimeters).
- **Quick Presets & Reset**: Instant "Try Setosa", "Try Versicolor", and "Try Virginica" presets to test archetype flowers, along with a "Reset" button.
- **Accurate Classification Pipeline**: Uses a Scikit-Learn `Pipeline` combining `StandardScaler` and `LogisticRegression` for scale-invariant predictions.
- **Confidence Scores & Breakdown**: Displays an overall confidence match percentage along with ranked probability distributions for all three species:
  - *Iris Setosa*
  - *Iris Versicolor*
  - *Iris Virginica*
- **REST API Endpoint**: Programmatic predictions available via `POST /api/predict`.
- **Robust Error Handling**: Handles non-numeric and out-of-range values gracefully with friendly feedback banners.
- **Automated Test Suite**: Unit and route tests covering model inference and web endpoints.

---

## Project Structure

```
iris_app/
├── app.py              # Flask app factory, web routes & REST API
├── config.py           # Configuration settings, presets & feature mappings
├── model.py            # Thread-safe model loader & prediction logic
├── train_model.py      # ML pipeline training and evaluation script
├── iris.pkl            # Trained model pipeline artifact
├── requirements.txt    # Python dependencies
├── Procfile            # Deployment command for Render
├── .gitignore          # Git ignore rules
├── templates/          # Jinja2 HTML templates
│   ├── base.html       # Shared base layout with modern typography
│   ├── home.html       # Interactive input sliders & quick presets
│   ├── after.html      # Prediction result, confidence bars & botanical notes
│   └── 404.html        # Custom 404 page
└── tests/              # Automated unit and route tests
    ├── test_model.py   # Model inference tests
    └── test_routes.py  # Web routes and API endpoint tests
```

---

## Setup & Running (with Virtual Environment)

### 1. Clone the Repository
```bash
git clone https://github.com/santoshkkashyap25/iris_app.git
cd iris_app
```

### 2. Create and Activate Virtual Environment

- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

- **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model Pipeline
```bash
python train_model.py
```
This trains the `StandardScaler + LogisticRegression` pipeline, displays test accuracy and classification metrics, and saves `iris.pkl`.

### 5. Run the Application
```bash
python app.py
```
The application will start at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## REST API Usage

### `POST /api/predict`

Send a `POST` request with a JSON body:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

*(You can also pass a list of 4 values: `{"features": [5.1, 3.5, 1.4, 0.2]}`)*

#### Sample Response:
```json
{
  "success": true,
  "prediction": {
    "class_id": 0,
    "species": "Iris-setosa",
    "display_name": "Iris Setosa",
    "confidence": 0.9808,
    "confidence_percentage": "98.1%",
    "description": "Known for small petals and sepals. Native to Japan and Arctic regions. Often bright blue-purple or white."
  },
  "probabilities": {
    "Iris-setosa": 0.9808,
    "Iris-versicolor": 0.0192,
    "Iris-virginica": 0.0
  },
  "input_features": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
```

---

## Running Automated Tests

Run the full test suite using `unittest`:
```bash
python -m unittest discover -s tests -p "test_*.py"
```