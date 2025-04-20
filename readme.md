# Wine Price Prediction using NLP and Numerical Features

This project explores predicting the price of a bottle of wine using both its text review description (NLP) and numerical features like points.

## Goal

The primary goal is to build a model that leverages both text patterns in wine reviews and numerical data (like points awarded) and categorical features (like region) to predict the wine's price more accurately than using text alone.

## Dataset

The data comes from the [Wine Reviews dataset on Kaggle](https://www.kaggle.com/datasets/zynicide/wine-reviews). It contains around 130,000 reviews scraped from *WineEnthusiast*. For this project, we use the `winemag-data-130k-v2.csv` file.

Key columns used:
*   `description`: The text review of the wine.
*   `points`: The points awarded to the wine (numerical feature).
*   `price`: The price of the wine bottle (our target variable).

## Methods

### 1. Data Loading & Preprocessing

*   The dataset was loaded using the pandas library.
*   Rows containing missing values (NaN), particularly in the `price` and `points` columns, were removed to ensure data quality for training.
*   The `description` and `points` columns were selected as input features (X), and the `price` column as the target variable (y).
*   **Text Processing:**
    *   Each wine `description` was broken down into individual words (tokenization).
    *   A vocabulary was built from all the unique words in the descriptions.
    *   Each description was converted into a sequence of numerical tokens based on the vocabulary.
    *   Sequences were padded to a fixed length (`maxlen=60`) for uniform model input.
*   **Numerical Feature Processing:**
    *   The `points` column was selected as the numerical input.
    *   The `RobustScaler` was used to scale the numerical features.
*   **Categorical Feature Processing:**
    *  The `region` is one hot encoded as used as another input.
      
### 2. Model Architecture (Combined)

A multi-input neural network model was built using the Keras Functional API to handle both text and numerical data:

1.  **Text Input Branch:**
    *   **Input Layer:** Takes the padded numerical token sequences (`shape=(maxlen,)`).
    *   **Embedding Layer:** Converts token sequences into dense vector embeddings (size 40). `mask_zero=True` handles padding.
    *   **Convolutional Layers (Conv1D):** Two 1D convolutional layers identify local patterns in the embeddings.
    *   **Flatten Layer:** Flattens the convolutional output.
    *   **Dense Layer:** A fully connected layer processes the flattened text features.

2.  **Numerical Input Branch:**
    *   **Input Layer:** Takes the numerical `points` feature (`shape=(1,)`).
    *   **Dense Layer:** A fully connected layer processes the numerical feature.

3.  **Concatenation:** The outputs from the text branch's dense layer and the numerical branch's dense layer are combined (concatenated).

4.  **Combined Processing:**
    *   **Dense Layers:** Fully connected layers process the combined features. A dropout layer (rate 0.15) is included for regularization.
    *   **Output Layer:** A single dense neuron with a 'relu' activation function outputs the predicted price.

### 3. Training

*   The model was compiled using the Mean Squared Error (MSE) loss function and the Adam optimizer with a learning rate of 1e-4. Mean Absolute Error (MAE) was tracked as the primary evaluation metric.
*   The data (padded text sequences and points) was split into training (70%) and validation (30%) sets.
*   The model was trained for up to 50 epochs with a batch size of 32, feeding both text and numerical inputs simultaneously.
*   **Early Stopping:** Training stopped early if the validation loss didn't improve for 2 consecutive epochs.
*   **TensorBoard:** Callbacks were used to log training progress.

## Results

*   **Baseline MAE:** Predicting the average price yielded an MAE of approximately **$17.73**.
*   **NLP-Only Model MAE:** The model using only text descriptions achieved a validation MAE of approximately **$14.13**.
*   **Combined Model MAE:** The final model incorporating both text (`description`) and numerical (`points`) features achieved a significantly better validation MAE of approximately **$11.48** after 11 epochs. This demonstrates the value of adding the `points` feature alongside the text for price prediction. Training stopped early due to the EarlyStopping callback.

*(Note: MAE represents the average absolute difference between the predicted price and the actual price. A lower MAE is better.)*

## Future Improvements

*   **Advanced Text Cleaning:** Implement more sophisticated text cleaning (e.g., removing stop words, stemming/lemmatization).
*   **Pre-trained Embeddings:** Utilize pre-trained word embeddings (Word2Vec, GloVe, FastText, BERT embeddings) for the text branch.
*   **Numerical Feature Scaling:** Apply scaling (e.g., StandardScaler) to the `points` feature and any other numerical features added.
*   **Different Architectures:** Experiment with RNNs (LSTM/GRU) or Transformers for the text branch. Explore different ways to combine text and numerical features (e.g., different concatenation points, attention mechanisms).
*   **Hyperparameter Tuning:** Systematically tune parameters for both branches and the combined layers (embedding size, convolutional filters, dense units, dropout, learning rate, etc.).
*   **Include More Features:** Incorporate other relevant features from the dataset (e.g., `country`, `variety`, `winery` - potentially after encoding them appropriately) into the numerical branch or as separate embedding inputs.
