# IOAI TST Kazakhstan - Day 4: Masked Word Position Prediction

This repository contains the solution for the fourth day's problem of the IOAI TST (Team Selection Test) in Kazakhstan. The task was to predict the original position of a masked word within a given sentence.

**Kaggle Competition Link:** [https://www.kaggle.com/competitions/kz-tst-day-4](https://www.kaggle.com/competitions/kz-tst-day-4)

## Table of Contents

  - [Problem Description](https://www.google.com/search?q=%23problem-description)
  - [Evaluation Metric](https://www.google.com/search?q=%23evaluation-metric)
  - [Solution Overview](https://www.google.com/search?q=%23solution-overview)
  - [Code Description](https://www.google.com/search?q=%23code-description)
  - [Results](https://www.google.com/search?q=%23results)
  - [Dependencies](https://www.google.com/search?q=%23dependencies)
  - [Usage](https://www.google.com/search?q=%23usage)

## Problem Description

The challenge involved natural language processing: given a sentence with one word removed (masked), the goal was to identify the original numerical index (position) of the missing word within that sentence.

## Evaluation Metric

The primary evaluation metric for this competition was **Accuracy**.

Our solution achieved an **Accuracy of 0.53677**.

## Solution Overview

The approach leverages a pre-trained multilingual BERT model for sequence understanding and classification. Key steps include:

1.  **Text Cleaning & Filtering:** Robust cleaning of raw text data to remove invalid characters, normalize whitespace, and filter out low-quality or short sentences.
2.  **Training Data Generation:** For each clean sentence, multiple training examples are created by randomly masking one word and using its original index as the target label.
3.  **Tokenization:** Utilizing a `BertTokenizer` to convert text into token IDs suitable for BERT.
4.  **Model Architecture:** A custom PyTorch model built on top of `bert-base-multilingual-cased`, with a linear classification layer predicting the word's position.
5.  **Training:** Fine-tuning the BERT-based model using Cross-Entropy Loss.
6.  **Inference:** Predicting the word positions for the test set.
7.  **Heuristic Post-processing:** Applying a simple heuristic to refine predictions: if a masked sentence starts with a lowercase letter, the predicted position is set to 0 (implying the first word was masked and it should have been capitalized).
8.  **Submission Generation:** Formatting the predictions into the required CSV format.

## Code Description

The provided Python script `batyr-yerdenov-3.ipynb` (despite the name, it's for Day 4) details the implementation:

  * **`seed_everything(seed=42)`:** Function to ensure reproducibility across runs.
  * **Constants & Device:** `tqdm.pandas()` for progress bars, `DEVICE` set to 'cuda' if available.
  * **Text Cleaning Functions (`clean_text`, `is_valid_char`, `is_clean_sentence`):**
      * `BAD_CHARS`: A set of undesirable characters to remove.
      * `is_valid_char`: Checks if a character is valid (e.g., Cyrillic, space, punctuation).
      * `clean_text`: Removes `BAD_CHARS`, normalizes punctuation and whitespace.
      * `is_clean_sentence`: Filters sentences based on length, capitalization patterns, repeated characters, short initial words, and vowel count to ensure quality.
  * **Data Loading:** Reads `train.csv` and `public_test.csv`.
  * **Initial Sentence Preprocessing:** Applies `clean_text` and `is_clean_sentence` to extract a clean set of sentences from the raw training data.
  * **Training Example Generation (`generate_n_masks`):**
      * Takes a clean sentence and generates `n` (default 3) masked versions.
      * Each masked version has one word randomly removed, and the index of the removed word is stored as the label.
  * **Dataset Splitting:** Splits the clean sentences into `train_sentences` and `val_sentences`.
  * **Dataset Creation:** Populates `train_data` and `val_data` lists by applying `generate_n_masks` to the respective sentence sets, then converts them to Pandas DataFrames.
  * **Tokenizer and Model Definition:**
      * `tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")`: Loads the pre-trained multilingual BERT tokenizer.
      * **`MaskedSentenceDataset` class:** A custom PyTorch `Dataset` to prepare input IDs, attention masks, and labels for masked sentences.
      * **`BertPositionModel` class:**
          * Initializes `BertModel.from_pretrained("bert-base-multilingual-cased")`.
          * Uses a dropout layer for regularization.
          * Employs a linear `classifier` layer to predict the word index (assuming a maximum sentence length, output size is 25). The `[CLS]` token's embedding is used as the sentence representation.
  * **Model Initialization & Optimization:**
      * `model = BertPositionModel().to(DEVICE)`: Moves the model to GPU if available.
      * `optimizer = AdamW(model.parameters(), lr=2e-5)`: AdamW optimizer, suitable for Transformer models.
      * `loss_fn = nn.CrossEntropyLoss()`: Loss function for multi-class classification.
  * **Training Loop:**
      * Iterates for a defined number of `EPOCHS`.
      * **Training Phase:** Sets model to `train()` mode, iterates through `train_loader`, performs forward pass, calculates loss, backpropagates, and updates weights.
      * **Validation Phase:** Sets model to `eval()` mode, iterates through `val_loader`, calculates predictions, and computes `accuracy_score`.
  * **Inference on Test Set:**
      * Applies `clean_text` to the test `masked_sentence` column.
      * **`TestDataset` class:** A simplified `Dataset` for test data, similar to `MaskedSentenceDataset` but without labels.
      * Sets model to `eval()` mode.
      * Iterates through `test_loader`, performs forward pass (without gradient calculation), and collects predictions.
  * **Heuristic Post-processing:**
      * Checks if the `masked_sentence_clean` in the test set starts with a lowercase letter.
      * If it does, the `word_index` prediction is forced to 0, otherwise, the model's prediction is used. This handles cases where the first word was masked and it should have been capitalized in the original sentence.
  * **Submission Generation:** Creates a `submission.csv` DataFrame with `ID` and `word_index` columns and saves it.

## Dependencies

  * `pandas`
  * `numpy`
  * `re`
  * `unicodedata`
  * `tqdm`
  * `scikit-learn`
  * `torch`
  * `transformers`

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn torch transformers tqdm
```

## Usage

1.  **Download the data:** Obtain `train.csv` and `public_test.csv` from the competition page ([https://www.kaggle.com/competitions/kz-tst-day-4](https://www.kaggle.com/competitions/kz-tst-day-4)) and place them in the specified Kaggle input directory (`/kaggle/input/tst-day-4/`). If running locally, adjust the paths in the script accordingly.
2.  **Run the Jupyter Notebook:** Open and run the `batyr-yerdenov-3.ipynb` notebook.
3.  **Generate Submission:** The script will automatically generate a `submission.csv` file in the same directory where the notebook is executed. This file will contain the `ID` and the predicted `word_index` for each test example.

-----
