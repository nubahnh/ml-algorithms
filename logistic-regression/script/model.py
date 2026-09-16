import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv(r'E:\TKCL\Projects\ML\ml-algorithms\logistic-regression\dataset\Social_Network_Ads.csv')

print(df.head())
print(df.info())

X = df.drop(columns=["Purchased", "User ID"])

# Convert Gender into numbers
# Female = 0
# Male = 1
X = pd.get_dummies(X, columns=["Gender"], drop_first=True)

y = df["Purchased"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y
)


scaler = StandardScaler() # feature scaling
numerical_columns = ["Age", "EstimatedSalary"]

X_train[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["0", "1"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()


print("\nClassification Report:")
print(classification_report(y_test, y_pred))

y_probability = model.predict_proba(X_test)[:, 1]

print("\nPrediction Probabilities:")
print(y_probability)