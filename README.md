# Feedback Prize - Predicting Effective Arguments

This project aims to predict the effectiveness of discourse elements in student essays for the Kaggle competition "[Feedback Prize - Predicting Effective Arguments](https://www.kaggle.com/competitions/feedback-prize-effectiveness)". It involves an iterative process of data analysis, model training, and prediction generation, evolving from an initial baseline to a more advanced approach.

## Project Structure

```
feedback_prize_effectiveness/
├── notebooks/                      # Jupyter notebooks for different stages
│   ├── eda.ipynb                   # Exploratory Data Analysis
│   ├── inference_bert.ipynb        # Kaggle submission for BERT
│   └── inference_longformer.ipynb  # Kaggle submission for Longformer
│   └── training_bert.ipynb         # BERT training and validation
│   └── training_longformer.ipynb   # Longformer training and validation
├── outputs/                        # Generated outputs
│   ├── result.png                  # Kaggle evaluation metrics
│   ├── submission.csv              # Sample submission file
│   └── wandb/                      # Local Weights & Biases logs
├── models/                         # Saved model artifacts (ignored by .gitignore)
├── data/                           # Raw competition data (ignored by .gitignore)
├── .gitignore                      # Specifies intentionally untracked files
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## Setup and Usage

1.  **Clone the repository.**
2.  **Data:** Download the competition data from Kaggle and place it into the `data/feedback-prize-effectiveness/` directory.
3.  **Environment:** Create a Python virtual environment and install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run Notebooks:** Execute the Jupyter notebooks in the `notebooks/` directory:
    * `eda.ipynb` for data exploration.
    * `training_bert` to train a single BERT model with simple train-val split
    * `inference_bert` to run inference with BERT
    * `training_longformer.ipynb` to train the Longformer model using GroupKFold.
    * `inference_longformer.ipynb` to generate predictions using an ensemble of the trained fold models.
5.  **Weights & Biases:** This project uses `wandb` for experiment tracking. Ensure you are logged in (`wandb login`) or have your API key configured.

## Approach Evolution & Key Technologies

The project started with a `bert-base-uncased` model. EDA revealed that essays were often longer than BERT's standard 512-token limit. This led to several key improvements:

* **Model:** Switched to `allenai/longformer-base-4096` to handle longer sequences (up to 1024 tokens).
* **Input Formulation:** Provided richer context by formatting input as: `[CLS] type [SEP] text [SEP] context_before [SEP] context_after [SEP]`.
* **Validation:** Implemented `GroupKFold` cross-validation (5 folds, grouped by `essay_id`) for more robust evaluation.
* **Training Optimization:** Used Automatic Mixed Precision (AMP) and 2-step gradient accumulation to accelerate Longformer training.
* **Submission:** Ensembled predictions by averaging the outputs from the best model of each of the 5 folds.

**Core Technologies:**
* Python
* Pandas, NumPy
* PyTorch
* Hugging Face Transformers (for `BERT` and `Longformer`)
* Scikit-learn
* Weights & Biases

## Results Summary

The iterative improvements resulted in a significant performance gain on the Kaggle private leaderboard:

* **Initial BERT model:** Private Score: 0.73063, Public Score: 0.75670.
* **Improved Longformer Ensemble:** Private Score: **0.65826**, Public Score: 0.65714.

This demonstrates the effectiveness of adapting the model architecture to data characteristics, refining input context, using robust validation, and ensembling.
