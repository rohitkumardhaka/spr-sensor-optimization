import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# load dataset
data = pd.read_excel("Material Database.xlsx")

# remove spaces in column names
data.columns = data.columns.str.strip()

print("Column names:")
print(list(data.columns))

print("Dataset preview:")
print(data.head())

# fill missing values
data = data.fillna("Unknown")

# define target column
target = "Optimized thickness"

# separate target
y = data[target]

# remove target from inputs
X = data.drop(target, axis=1)

# convert categorical features
X = pd.get_dummies(X)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestRegressor(n_estimators=100)

# train
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)

# evaluate
mse = mean_squared_error(y_test, predictions)

print("\nModel training complete")
print("Mean Squared Error:", mse)