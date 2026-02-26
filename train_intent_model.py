# train_intent_model.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import joblib

TRAIN_DATA = [
    # greeting
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("hey", "greeting"),
    ("good morning", "greeting"),
    ("good afternoon", "greeting"),
    ("good evening", "greeting"),
    ("hi there", "greeting"),
    ("hey bot", "greeting"),
    ("hello bot", "greeting"),
    ("greetings", "greeting"),
    ("nice to meet you", "greeting"),
    ("howdy", "greeting"),

    # ask user name
    ("what is my name","ask_user_name"),
    ("do you know my name", "ask_user_name"),
    ("who am i", "ask_user_name"),
    ("tell me my name", "ask_user_name"),
    ("whats my name", "ask_user_name"),
    ("can you tell me my name","ask_user_name"),
    ("do you remember my name", "ask_user_name"),
    ("what did i tell you my name is", "ask_user_name"),
    ("my name?", "ask_user_name"),
    ("who am i again", "ask_user_name"),

    # set user name
    ("my name is eva", "set_user_name"),
    ("my name is maria", "set_user_name" ),
    ("i am Eva", "set_user_name"),
    ("i'm eva", "set_user_name"),
    ("call me eva", "set_user_name"),
    ("you can call me eva", "set_user_name"),#
    ("call me maria", "set_user_name"),
    ("i'm maria", "set_user_name"),
    ("i am maria", "set_user_name"),
    ("please call me maria", "set_user_name"),
    ("you can call me maria", "set_user_name"),

    # exit
    ("bye", "exit"),
    ("exit", "exit"),
    ("quit", "exit"),
    ("goodbye", "exit"),
    ("see you", "exit"),
    ("see you later", "exit"),
    ("i have to go", "exit"),
    ("talk to you later", "exit"),
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

    # pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            lowercase=True,
            strip_accents="unicode"
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )),
    ])

    # train
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    # ------------ threshold sweep -------------

    classes = list(model.classes_)
    pred_idx = proba.argmax(axis=1)
    pred_label = np.array(classes)[pred_idx]
    pred_conf = proba.max(axis=1)
    y_test_arr = np.array(y_test)

    print("\n=== THRESHOLD SWEEP ===")
    print("t    accepted coverage   accuracy_on_accepted")

    for t in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        accepted = pred_conf >= t
        accepted_n = int(accepted.sum())
        coverage = accepted.mean()

        if accepted_n == 0:
            print(f"{t:.2f} {accepted_n:8d} {coverage:8.2f}  -")
            continue

        acc_t =accuracy_score(y_test_arr[accepted], pred_label[accepted])
        print(f"{t:.2f}  {accepted_n:8d} {coverage:8.2f} {acc_t:8.2f}")

    # predict probabilities on text set
    proba = model.predict_proba(X_test)

    # binarize labels for One-vs-Rest ROC-AUC
    classes = list(model.classes_)
    y_test_bin = label_binarize(y_test, classes=classes)

    auc_macro = roc_auc_score(y_test_bin, proba, average="macro", multi_class="ovr")
    print(f"ROC-AUC (MACRO, ovr): {auc_macro:.3f}")

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