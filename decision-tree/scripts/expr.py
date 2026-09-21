import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


df = pd.read_csv(r"E:\TKCL\Projects\ML\ml-algorithms\decision-tree\dataset\palmerpenguins_extended.csv")

print("Original shape:", df.shape)
print(df.head())
print(df.columns)
print("\nDuplicates:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

features = [
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
    "diet",
    "life_stage",
    "health_metrics",
    "year"
]

target = "species"

data = df[[target] + features].dropna()

print("\nShape after removing missing values:", data.shape)

X = data[features]
y = data[target]

print("\nTarget distribution:")
print(y.value_counts())

categorical_features = [
    "island",
    "sex",
    "diet",
    "life_stage",
    "health_metrics"
]

X = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=False
)

print("\nEncoded features:")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print("\nTraining Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
).plot()

plt.title("Decision Tree Confusion Matrix")
plt.show()

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(10, 8))

importance.plot(kind="barh")

plt.title("Decision Tree Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree for Penguin Species")
plt.show()
