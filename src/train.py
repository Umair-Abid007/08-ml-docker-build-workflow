import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from preprocess import preprocess_data

def train_model():
    # Load and preprocess data
    df = preprocess_data("data/sample.csv")
    X = df[["age"]]
    y = df["salary"]

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict and calculate metric
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"Training completed. MSE: {mse:.2f}")

if __name__ == "__main__":
    train_model()
