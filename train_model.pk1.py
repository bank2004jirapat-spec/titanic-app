import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# โหลดข้อมูล Titanic
df = pd.read_csv("titanic.csv")

# เลือก features ให้ตรงกับ app.py
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

# แปลงข้อมูลตัวหนังสือเป็นตัวเลข
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X["Embarked"] = X["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# แก้ค่าว่าง
X = X.fillna(0)

# แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# สร้างโมเดล
model = RandomForestClassifier()
model.fit(X_train, y_train)

# บันทึกโมเดลใหม่
joblib.dump(model, "titanic_model.pkl")

print("เทรนเสร็จแล้ว ✅")
