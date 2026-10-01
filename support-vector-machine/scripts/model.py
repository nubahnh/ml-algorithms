import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"E:\TKCL\Projects\ML\ml-algorithms\support-vector-machine\datasets\diabetes.csv")

print(df.info)
print(df.columns.tolist())
print(df.head())
print(df.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,confusion_matrix,precision_score,recall_score,f1_score)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("0 = No Diabetes")
print("1 = Diabetes")

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel="rbf",random_state=42)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

precision = precision_score(y_test, y_pred)
print('\nPrecision: ', precision)

recall = recall_score(y_test, y_pred)
print('\nRecall: ', recall)

f1 = f1_score(y_test,y_pred)
print('\nF-1 Score: ',f1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap="Purples")

plt.title("SVM - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["No Diabetes", "Diabetes"])
plt.yticks([0, 1], ["No Diabetes", "Diabetes"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],ha="center",va="center")

plt.colorbar()

plt.show()


