# Feedback Prize - Predicting Effective Arguments

This project aims to predict the effectiveness of discourse elements in student essays for the Kaggle competition "[Feedback Prize - Predicting Effective Arguments](https://www.kaggle.com/competitions/feedback-prize-effectiveness)". It involves data analysis, training a transformer-based model, and generating predictions.

## Project Structure

feedback_prize_effectiveness/├── notebooks/                  # Jupyter notebooks for different stages│   ├── eda.ipynb               # Exploratory Data Analysis│   ├── training_and_validation.ipynb # Model training and validation│   └── prediction_and_submission.ipynb # Generating predictions for submission├── outputs/                    # Generated outputs│   ├── submission.csv          # Sample submission file│   └── wandb/                  # Local Weights & Biases logs├── models/                     # Saved model artifacts│   └── best_model/             # Best trained model and tokenizer├── data/                       # Raw competition data (ignored by .gitignore)├── .gitignore                  # Specifies intentionally untracked files├── README.md                   # This file└── requirements.txt            # Python dependencies (you should create this)
## Setup and Usage

1.  **Clone the repository.**
2.  **Data:** Download the competition data from Kaggle and place it into the `data/feedback-prize-effectiveness/` directory.
3.  **Environment:** Create a Python virtual environment and install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You'll need to generate `requirements.txt` using `pip freeze > requirements.txt`)*
4.  **Run Notebooks:** Execute the Jupyter notebooks in the `notebooks/` directory, typically starting with `eda.ipynb`, then `training_and_validation.ipynb`, and finally `prediction_and_submission.ipynb`.
5.  **Weights & Biases:** This project uses `wandb` for experiment tracking. Ensure you are logged in (`wandb login`) or have your API key configured.

## Key Technologies

* Python
* Pandas, NumPy
* PyTorch
* Hugging Face Transformers (for `bert-base-uncased`)
* Scikit-learn
* Weights & Biases

## Results Summary

* **Local Validation (Best):** Log Loss of ~0.5725.
* **Kaggle (Late Submission):** Public Score 0.75670, Private Score 0.73063.

The model demonstrates a good baseline for the task, with further improvements possible through more extensive training, hyperparameter tuning, and advanced validation techniques.
