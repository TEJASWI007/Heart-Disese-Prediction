# Heart Disease Prediction App

Welcome to the Heart Disease Prediction App! This application allows users to predict their heart health based on various factors using machine learning models.

## Features

- Predicts heart disease risk based on user input.
- Provides detailed explanations for the prediction.
- Easy-to-use interface.

## Usage

1. **Input Details**: Fill in the required details about age, gender, chest pain, blood pressure, cholesterol, etc.
2. **Predict**: Click on the "Predict My Heart Health" button to get the prediction.
3. **View Results**: See the prediction result and any warnings or recommendations provided.
4. **More Information**: Check the box to request more information about the disease.
5. **Enter OpenAI Key**: If requested, enter your OpenAI API key.
6. **Submit**: Click the "Submit" button to get more detailed explanations.

## Installation

1. Clone this repository-https://github.com/AIOnGraph/Heart-Disease-Prediction.git
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app using `streamlit run app.py`.

## Files

- `decision_tree_model.joblib`: Joblib model file for the Decision Tree model.
- `Logistic_regression_model.joblib`: Joblib model file for the Logistic Regression model.
- `KNN_model.joblib`: Joblib model file for the KNN model.
- `naive_bayes_model.joblib`: Joblib model file for the Naive Bayes model.
- `svc_model.joblib`: Joblib model file for the SVM model.
- `deep_learning_model.joblib`: Joblib model file for the Deep Learning model.
- `scaler.joblib`: Joblib file for the scaler used in model preprocessing.
- `heartimage.png`: Image file used for background in the Streamlit app.

## Libraries Used

- Streamlit
- Joblib
- Pandas
- dotenv
- base64
- langchain
- time
- openai

## Contributing
Contributions are welcome! If you want to contribute to this project, feel free to open an issue or create a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

