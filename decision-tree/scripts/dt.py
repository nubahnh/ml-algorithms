import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score,recall_score


df = pd.read_csv(
    r"E:\TKCL\Projects\ML\ml-algorithms\decision-tree\dataset\palmerpenguins_extended.csv"
)

print(df.columns.tolist())
print(df.head())
print(df.isnull().sum())

data = df[
    [
        "species",
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]
].dropna()

X = data[
    [
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]
]

y = data["species"]

X = pd.get_dummies(
    X,
    columns=["island"]
)

X = X.rename(columns={
    "island_Biscoe": "B",
    "island_Dream": "D",
    "island_Torgensen": "T"
})

print("\nEncoded features:")
print(X.head())

print("\nFeature names:")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("\nAccuracy:", accuracy)
print("\nPrecision:", precision)
print("\nRecall: ", recall)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(16, 9))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True
)

plt.title("Decision Tree - Penguin Species")
plt.show()


importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(10, 6))

importance.plot(kind="barh")

plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()