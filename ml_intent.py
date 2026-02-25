import joblib

class IntentModel:
    def __init__(self, model_path: str = "intent_model.joblib"):
        self.model = joblib.load(model_path)

    def predict(self, text: str) -> tuple[str, float]:
        """
        Returns: (intent_label, confidence)
        confidence is max probability from predict_proba
        """
        text = text.strip()

        if not text:
            return "unknown", 0.0

        proba = self.model.predict_proba([text])[0]
        best_idx = int(proba.argmax())
        intent = self.model.classes_[best_idx]
        confidence = float(proba[best_idx])
        return intent, confidence