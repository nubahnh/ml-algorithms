
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"E:\TKCL\Projects\ML\ml-algorithms\naive-bayes\datasets\Dry_Bean_Dataset-Dry_Beans_Dataset.csv")
print(df.head())
print(df.shape)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score,precision_score,recall_score, f1_score,confusion_matrix,ConfusionMatrixDisplay,)

X = df.drop("Class", axis=1)
y = df["Class"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,test_size=0.20,random_state=42,stratify=y_encoded)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GaussianNB()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=label_encoder.classes_)

disp.plot(xticks_rotation=45)

plt.title("Naive Bayes - Dry Bean Classification")
plt.tight_layout()
plt.show()

import random
bean_number = random.randint(0, len(X_test_scaled) - 1) # Select a random test sample

probabilities = model.predict_proba(X_test_scaled[bean_number:bean_number + 1])[0]

bean_names = label_encoder.classes_

for bean, probability in zip(bean_names, probabilities):
    print(f"{bean}: {probability:.2%}")

predicted_bean = bean_names[probabilities.argmax()]
print("\nPredicted Bean:", predicted_bean)

plt.figure(figsize=(10, 5))

plt.bar(bean_names, probabilities)
plt.title("Naive Bayes Prediction Probabilities")
plt.xlabel("Bean Class")
plt.ylabel("Probability")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()