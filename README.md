# LSTM-Based Sentiment Analysis

## Aim

To implement a Long Short-Term Memory (LSTM) neural network for sentiment analysis and classify movie reviews as positive or negative based on learned linguistic patterns.

## Objectives

* Understand the architecture and working of LSTM networks.
* Perform text preprocessing and tokenization.
* Convert textual data into numerical sequences.
* Build and train an LSTM-based sentiment classification model.
* Evaluate the model using appropriate performance metrics.
* Predict the sentiment of new movie reviews.

## Technologies Used

* Python
* TensorFlow and Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Google Colab
* GitHub

## Dataset

The project uses the IMDB Movie Reviews dataset available through `tensorflow.keras.datasets.imdb`.

The dataset contains 50,000 movie reviews:

* 25,000 training reviews
* 25,000 testing reviews
* Positive sentiment label: `1`
* Negative sentiment label: `0`

The vocabulary is limited to the 10,000 most frequently occurring words. Reviews are converted into integer sequences and padded or truncated to a fixed length of 200 tokens.

## Methodology

### Part A: Dataset Preparation

1. Import the required libraries.
2. Load and explore the IMDB dataset.
3. Examine sample reviews and sentiment labels.
4. Convert reviews into numerical sequences.
5. Apply sequence padding to obtain equal-length inputs.
6. Split the training data into training and validation sets.

### Part B: LSTM Model Implementation

The model consists of the following layers:

1. **Embedding Layer:** Converts integer token IDs into dense vector representations.
2. **LSTM Layer:** Learns sequential patterns and contextual dependencies in reviews.
3. **Dropout Layer:** Helps reduce overfitting during training.
4. **Dense Output Layer:** Uses sigmoid activation to predict the probability of positive sentiment.

The model is compiled using the Adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric.

### Part C: Training and Evaluation

The model is trained using the prepared training dataset and validated after each epoch. Its performance is evaluated using:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss
* Test accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Accuracy and loss graphs are plotted to observe the model's learning progress.

### Part D: Sentiment Prediction

New movie-review sentences are preprocessed using the same vocabulary and sequence length. The trained model predicts a sentiment probability, which is converted into a positive or negative label using a threshold of 0.5. The predicted sentiment is compared with the expected sentiment to examine the model's performance on sample reviews.

## Model Architecture

| Layer     | Configuration                         | Purpose                     |
| --------- | ------------------------------------- | --------------------------- |
| Embedding | 10,000 vocabulary size, 64 dimensions | Word representation         |
| LSTM      | 64 units                              | Sequential feature learning |
| Dropout   | 0.5                                   | Regularization              |
| Dense     | 1 unit, sigmoid activation            | Binary classification       |

## How to Run

1. Open the notebook in Google Colab.
2. Run the cells in order, from Part A to Part D.
3. Allow the model to train and complete the evaluation.
4. Observe the accuracy and loss plots.
5. Review the confusion matrix and classification metrics.
6. Test the model using new movie reviews.
7. Save the notebook and upload it to GitHub.

## Results

The trained LSTM model classifies movie reviews into positive and negative sentiment categories. Its performance is measured using test accuracy, precision, recall, F1-score, and the confusion matrix. Training and validation curves help analyze learning behavior and identify possible overfitting.

**Note:** Insert the actual test accuracy, precision, recall, and F1-score obtained from the notebook after running the experiment.

## Conclusion

The experiment demonstrates how LSTM networks can process sequential textual data and learn contextual relationships between words for sentiment classification. Text preprocessing, numerical encoding, embedding, and sequence padding prepare the data for training. The evaluation metrics and prediction examples help assess the model's effectiveness in classifying movie reviews.

## Repository Contents

* `LSTM_Sentiment_Analysis_ROLLNO.ipynb` — Google Colab notebook containing the implementation and outputs.
* `README.md` — Project description, methodology, and instructions.
* `lstm_sentiment_model.keras` — Saved trained model, if included.
* `imdb_word_index.json` — Vocabulary mapping, if included.

## Author

* Name: YOUR_NAME
* Roll Number: YOUR_ROLL_NUMBER
