import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)


df = pd.read_csv(r'E:\TKCL\Projects\ML\ml-algorithms\k-nearest neighbours\dataset\data.csv')

print(df.head())
print(df.shape)
df = df.drop("id", axis=1)
df = df.drop("Unnamed: 32", axis=1)
print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0})


X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

print("\nNumber of features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(  X,y,test_size=0.2,random_state=42,stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_values = range(1, 21) # for finding what value of k gives best result
cv_scores = []

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    cv_scores.append(scores.mean())

best_k = k_values[
    cv_scores.index(max(cv_scores))
]

print("\nBest K:", best_k)


knn_before = KNeighborsClassifier(n_neighbors=best_k)
knn_before.fit(X_train_scaled,y_train)


y_pred_before = knn_before.predict(X_test_scaled)
accuracy_before = accuracy_score(y_test,y_pred_before)
precision_before = precision_score(y_test,y_pred_before)
recall_before = recall_score(y_test,y_pred_before)
f1_before = f1_score(y_test,y_pred_before)

print("\nBEFORE FEATURE SELECTION")

print("Number of Features:", X.shape[1])
print("Best K:", best_k)

print("\nEvaluation Metrics:")
print("Accuracy :", accuracy_before)
print("Precision:", precision_before)
print("Recall   :", recall_before)
print("F1 Score :", f1_before)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_before))

selector = SelectKBest(score_func=f_classif,k=10) #Feature Selection
X_train_selected = selector.fit_transform(X_train,y_train)
X_test_selected = selector.transform(X_test)
selected_features = X.columns[
    selector.get_support()
]

print("\nSELECTED FEATURES")

for feature in selected_features:
    print(feature)

scaler_selected = StandardScaler() # Scaling
X_train_selected_scaled = scaler_selected.fit_transform(X_train_selected)

X_test_selected_scaled = scaler_selected.transform(X_test_selected)
cv_scores_selected = []

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model,X_train_selected_scaled,y_train,cv=5,scoring="accuracy")
    cv_scores_selected.append(scores.mean())

best_k_selected = k_values[cv_scores_selected.index(max(cv_scores_selected))]

print("\nBest K after feature selection:", best_k_selected)

knn_after = KNeighborsClassifier(n_neighbors=best_k_selected)

knn_after.fit(X_train_selected_scaled,y_train)

y_pred_after = knn_after.predict(X_test_selected_scaled)

#Evaluation AFTER Feature Selection
accuracy_after = accuracy_score(y_test,y_pred_after)

precision_after = precision_score(y_test,y_pred_after)

recall_after = recall_score(y_test,y_pred_after)

f1_after = f1_score(y_test,y_pred_after)

print("\nAFTER FEATURE SELECTION")

print("Number of Features:", X_train_selected.shape[1])
print("K value selected:", best_k_selected)

print("\nEvaluation Metrics:")
print("Accuracy :", accuracy_after)
print("Precision:", precision_after)
print("Recall   :", recall_after)
print("F1 Score :", f1_after)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_after))



comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Before (30 Features)": [
        accuracy_before,
        precision_before,
        recall_before,
        f1_before
    ],

    "After (10 Features)": [
        accuracy_after,
        precision_after,
        recall_after,
        f1_after
    ]
})


