

from pathlib import Path
import joblib
from flask import Flask, render_template, request


def create_app() -> Flask:
    app = Flask(__name__)

    # Paths to the trained artefacts.  They live in the same directory as this
    # file, so we construct paths relative to ``app.py``.
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "model.pkl"
    vectorizer_path = base_dir / "vectorizer.pkl"

    # Load the trained model and vectoriser.  These are only loaded once
    # when the app is created to avoid reloading on each request.
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Model or vectoriser file not found. Have you run train_model.py?"
        )
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    @app.route("/")
    def index():
        """Render the landing page with the input form."""
        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        """Handle form submission and return the prediction result."""
        news_text = request.form.get("news")
        if not news_text:
            return render_template("result.html", prediction="Please enter some text.")
        # Transform the input text into the same feature space used during training
        transformed = vectorizer.transform([news_text])
        pred_label = model.predict(transformed)[0]
        return render_template("result.html", prediction=pred_label)

    return app


if __name__ == "__main__":
    # Create the app and run it.  In development mode ``debug=True`` enables
    # hot reloads and an interactive debugger.  In production, set debug
    # to False and use a WSGI server such as gunicorn.
    flask_app = create_app()
    flask_app.run(debug=True, port=5005)