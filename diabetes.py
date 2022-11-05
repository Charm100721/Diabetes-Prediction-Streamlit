import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("diabetes.csv")
features = data.drop(columns="Outcome", axis=1)
x= features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
Result = data["Outcome"]
y = Result

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
train_accuracy_score = accuracy_score(y_test, y_pred)

print(train_accuracy_score)
x_test_prediction = model.predict(x_test)
test_accuracy_score = accuracy_score(x_test_prediction, y_test)
print(f"Accuracy Score: {test_accuracy_score}")

st.title("Diabetes Prediction")

pregnancies = st.number_input("NUMBER OF PREGNANCIES", 0,17,1)
glucose = float(st.number_input("GLUCOSE LEVEL", 0,199,128))
BP = float(st.number_input("BLOOD PRESSURE LEVEL", 0,122, 88))
skin_thickness = float(st.number_input("SKIN THICKNESS", 0,99, 39))
insulin = st.number_input("INSULIN LEVEL",0,846, 110)
BMI = float(st.number_input("BODY MASS INDEX", 0,67, 37))
DPF = float(st.number_input("DIABETES PEDIGREE FUNCTION", 0,3, 1))
age = st.number_input("AGE", 21,81, 37)

def predict():
    features = np.array([pregnancies, glucose, BP, skin_thickness, insulin, BMI, DPF, age])
    X = pd.DataFrame([features])
    prediction = model.predict(X)

    if prediction[0] == 1:
        st.error("The patient is diagnosed **WITH DIABETES**.")
    else:
        st.success("The patient is **HEALTHY**.")
    


predict_button = st.button("PREDICT")
if predict_button == True:
    predict()


st.sidebar.title("Diabetes Awareness")
st.sidebar.write("An estimated 537 million adults (20-79 years old) will have diabetes in 2021. By 2030, there will be 643 million diabetics worldwide, and by 2045, there will be 783 million. In nations with low and intermediate incomes, 3 out of 4 adults with diabetes reside.")

st.sidebar.write("Patients with uncontrolled diabetes have high blood glucose levels. Since blood is the source of all biologic fluids, changes in blood composition also have an impact on bodily fluids. Considering saliva is a vital bodily fluid, high blood sugar levels also result in higher salivary glucose levels.")