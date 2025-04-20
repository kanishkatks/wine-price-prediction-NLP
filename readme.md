# Wine Price Prediction from Reviews and Numerical Features

## Project Goal

This project aims to predict the price of a wine bottle using its description and numerical features like points and region. We want to build a model that can provide an estimate of a wine's price based on textual and numerical data.

## Methods

1. **Data Acquisition:**
    * The project utilizes the Wine Reviews dataset from Kaggle.
    * The dataset contains information on wine descriptions, points, price, region, and other attributes.

2. **Data Preprocessing:**
    * The wine descriptions are processed using natural language processing techniques (NLP).
        * Convert text to word sequences.
        * Create a vocabulary of unique words.
        * Pad sequences to a fixed length for model input.
    * Numerical and categorical features are preprocessed:
        * Points are used directly.
        * Region is one-hot encoded to handle categorical values.
        * Numerical features are standardized using `StandardScaler`.

3. **Model Development:**
    * **NLP Model:**
        * An embedding layer converts words into numerical vectors.
        * Convolutional layers extract features from the text.
        * Dense layers process features and predict price.
    * **Numerical Features Model:**
        * Dense layers process numerical and categorical features to predict price.
    * **Combined Model:**
        * Combines outputs of NLP and numerical features models.
        * A final dense layer predicts the price based on the combined information.

4. **Model Training and Evaluation:**
    * Models are trained using the `winemag-data-130k-v2.csv` dataset.
    * Mean Squared Error (MSE) is used as the loss function.
    * Mean Absolute Error (MAE) is used as the evaluation metric.
    * Early stopping is implemented to prevent overfitting.

## Results

The combined model, leveraging both text and numerical features, achieves the best performance in predicting wine prices.

## Future Improvements

* Explore more advanced NLP techniques like BERT or transformers.
* Incorporate more features, such as winery and designation.
* Fine-tune hyperparameters to improve model accuracy.
* Deploy the model as a web application for real-time price prediction.
