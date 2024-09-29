import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


st.title("Sleep Quality Prediction")


csv_file = "Health_Sleep_Statistics.csv"  
df = pd.read_csv(csv_file)


st.subheader("Raw Data")
st.write(df)


df['Gender'] = df['Gender'].map({'m': 0, 'f': 1})


df['Bedtime'] = pd.to_datetime(df['Bedtime'], format='%H:%M').dt.hour * 60 + pd.to_datetime(df['Bedtime'], format='%H:%M').dt.minute
df['Wake-up Time'] = pd.to_datetime(df['Wake-up Time'], format='%H:%M').dt.hour * 60 + pd.to_datetime(df['Wake-up Time'], format='%H:%M').dt.minute

st.subheader("Processed Data")
st.write(df)


features = ['Age', 'Gender', 'Bedtime', 'Wake-up Time', 'Daily Steps', 'Calories Burned']
target = 'Sleep Quality'


X = df[features]
y = df[target]


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy * 100:.2f}%")


st.subheader("Rate Your Sleep Quality Between 1 to 10")


col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, step=1)

with col2:
    steps = st.number_input("Daily Steps", min_value=0, max_value=50000, step=1000)

with col3:
    calories = st.number_input("Calories Burned", min_value=0, max_value=10000, step=100)

with col1:
    gender = st.selectbox("Gender", ('Male', 'Female'))

with col2:
    bedtime = st.time_input("Bedtime")

with col3:
    wakeup_time = st.time_input("Wake-up Time")


input_data = {
    'Age': age,
    'Gender': 0 if gender == 'Male' else 1,
    'Bedtime': bedtime.hour * 60 + bedtime.minute,
    'Wake-up Time': wakeup_time.hour * 60 + wakeup_time.minute,
    'Daily Steps': steps,
    'Calories Burned': calories
}

input_df = pd.DataFrame([input_data])


input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = clf.predict(input_scaled)
    if(prediction[0]<4):
        type = 'Bad'
    elif(prediction<7):
        type = 'Average'
    else:
        type = 'Good'
    st.markdown(f"#### Predicted Sleep Quality: {prediction[0]} - {type}")