import streamlit as st
import joblib
import numpy as np

# โหลดโมเดล
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Prediction")

st.write("กรอกข้อมูลผู้โดยสารเพื่อทำนายการรอดชีวิต")

# รับค่าจากผู้ใช้
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
parch = st.number_input("Parents/Children", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# แปลงเพศเป็นตัวเลข
if sex == "Male":
    sex = 0
else:
    sex = 1

# แปลง Embarked เป็นตัวเลข
if embarked == "S":
    embarked = 0
elif embarked == "C":
    embarked = 1
else:
    embarked = 2

# ปุ่มทำนาย
if st.button("Predict"):

    # สร้าง input ให้ครบ 7 features
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success(f"🎉 Prediction: Survived")
    else:
        st.error(f"❌ Prediction: Not Survived")

    st.write(f"Probability of Survival: {probability[0][1]*100:.2f}%")
