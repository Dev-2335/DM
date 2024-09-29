# Sleep Quality Prediction App

This Streamlit-based web app predicts sleep quality based on user inputs like age, gender, bedtime, wake-up time, daily steps, and calories burned using a machine learning model. The app was built using Python, Pandas, and Scikit-learn.

## Features

- **Interactive User Input**: Users can input their personal details such as age, daily steps, calories burned, gender, bedtime, and wake-up time.
- **Data Preprocessing**: The app processes the bedtime and wake-up time inputs into numerical formats (minutes) and standardizes the features using `StandardScaler`.
- **Machine Learning Model**: The app uses a `RandomForestClassifier` trained on a dataset to predict sleep quality.
- **Accuracy Display**: Shows the accuracy of the trained model.
- **Sleep Quality Prediction**: Predicts the user's sleep quality and categorizes it as 'Bad', 'Average', or 'Good'.

## Dataset

The model is trained on a dataset called `Health_Sleep_Statistics.csv`. The dataset contains the following columns:
- `Age`
- `Gender` (male/female)
- `Sleep Quality` (Target variable: rating between 1 to 10)
- `Bedtime` (e.g., 22:30)
- `Wake-up Time` (e.g., 06:30)
- `Daily Steps`
- `Calories Burned`

## How to Run the App

### Prerequisites

- Python 3.8+
- Required Python libraries (listed in `requirements.txt`):
  - `streamlit`
  - `pandas`
  - `scikit-learn`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/sleep-quality-predictor.git
    cd sleep-quality-predictor
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

### Usage

1. After launching the app, you will be able to view and explore the raw and processed data used for prediction.
2. Input your details like age, bedtime, wake-up time, daily steps, calories burned, and gender.
3. Click the **Predict** button to get the predicted sleep quality and a rating based on the prediction.

## Model Details

- **Algorithm**: RandomForestClassifier
- **Accuracy**: The app displays the model's accuracy on the test data split (80/20 train-test).


