import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"E:\TKCL\Projects\ML\ml-algorithms\linear-regression\dataset\house_price_regression_dataset.csv")
print(df.shape)
print(df.info())
print(df.describe())
print(df.columns)
y = df["House_Price"]
X = df[[
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality"
]]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#print(y_pred)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE: {:.2f}".format(mean_absolute_error(y_test, y_pred)))
print("MSE: {:.2f}".format(mean_squared_error(y_test, y_pred)))
print("R2: {:.2f}".format(r2_score(y_test, y_pred)))



plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.show()