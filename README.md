# Supervised NLP intent classification system using TF-IDF and Logistic Regression with cross-validation, hyperparameter tuning and confidence thresholding 

This repository contains a small command-line chatbot that combines:
- Rule-based dialogue logic and
- Intent classification using ein ML model

The goal is to show a minimal und interpretable NLP pipeline (TF-IDF + Logistic Regression) for intent detection.

I try different schemas and I observe the changes in the classification.

## Structure

- main.py -> CLI
- chatbot.py 
- memory.py
- memory.json
- train.intent_model.py -> trains and saves the model

## Features

- Greets the user
- Responds to basic questions
- Remembers the user's name
- Saves memory between sessions
- Simple intent-based logic (no external APIs)

A confidence threshold is used to reduce wrong classification on short inputs.

## How it works

1. User input is preprocessed (basic normalization).
2. If the system is currently waiting for a name, the input is treated as the user's name.
3. Otherwise, the ML model predicts an intent and a confidence score.
4. if confidence is below a threshold,the chatbot falls back to a default response.
5. For name-realted intents, the chatbot reads/writes state via Memory and persists it in memory.json.

## Technologies used

- Python3
- JSON for simple data storage
- Git & GitHub

