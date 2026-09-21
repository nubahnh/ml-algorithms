import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix


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

for depth in [2, 3, 4, 5, 6, 7, 8, 10]:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Depth: {depth}, Accuracy: {accuracy:.4f}")



