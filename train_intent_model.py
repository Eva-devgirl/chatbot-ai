# train_intent_model.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

TRAIN_DATA = [
    # greeting
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("hey", "greeting"),
    ("good morning", "greeting"),
    ("good afternoon", "greeting"),
    ("good evening", "greeting"),

    # ask user name
    ("what is my name","ask_user_name"),
    ("do you know my name", "ask_user_name"),
    ("who am i", "ask_user_name"),
    ("tell me my name", "ask_user_name"),
    ("whats my name", "ask_user_name"),

    # set user name
    ("my name is eva", "set_user_name"),
    ("my name is maria", "set_user_name" ),
    ("i am Eva", "set_user_name"),
    ("i'm eva", "set_user_name"),
    ("call me eva", "set_user_name"),
    ("you can call me eva", "set_user_name"),

    # exit
    ("bye", "exit"),
    ("exit", "exit"),
    ("quit", "exit"),
    ("goodbye", "exit"),
]

def main():
    texts = [t for (t, label) in TRAIN_DATA]
    labels = [label for (t, label) in TRAIN_DATA]

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size = 0.3,
        random_state = 42,
        stratify = labels
    )

    # build model
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    # train
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # evaluate
    acc =accuracy_score(y_test, y_pred)
    print("\n=== EVALUATION ===")
    print(f"Accuracy: {acc:.3f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # save model
    joblib.dump(model, "intent_model.joblib")
    print("Saved model to intent_model.joblib.")

if __name__ == "__main__":
    main()