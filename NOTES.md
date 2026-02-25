### 'NOTES.md'
   
# Implementation Notes (Development Log)
- Chose a TF-IDF + Logistic Regression classifier because it is fast and suitable for small datasets.
- Added a confidence threshold to avoid confidently wrongintent predictions on short inputs.
- Observed that inputs like "how are you" can be misclassified when the training set is small; the threshold helps reducing the errors.
- Implemented a small Memory layer to persist simple state (e.g. name, awaiting_name) between session via memory.json.
- Tuned threshold (example: 0.6 -> 0.3) after some wrong outputs and the general behavior.
- Kept the dialogue routing quite simple. 
