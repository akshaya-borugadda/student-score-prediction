# student_marks.py (or put into a Jupyter cell)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle

# 1. Load data
df = pd.read_csv('student_scores.csv')
print("First rows:\n", df.head())

# 2. Visualize Hours vs Score
plt.figure(figsize=(6,4))
plt.scatter(df['Hours'], df['Scores'])
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.title('Hours vs Score')
plt.grid(True)
plt.show()

# 3. Prepare data
X = df[['Hours']]   # features must be 2D
y = df['Scores']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2  :", r2_score(y_test, y_pred))

# 7. Plot regression line with data
plt.figure(figsize=(6,4))
plt.scatter(X, y, label='Data points')
line_X = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
line_y = model.predict(line_X)
plt.plot(line_X, line_y, color='red', label='Regression line')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.show()

# 8. Predict for a new input
hours = 9.25
predicted_score = model.predict(np.array([[hours]]))[0]
print(f"Predicted score for studying {hours} hours: {predicted_score:.2f}")

# 9. Save the trained model (optional, useful to load later)
with open('student_score_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved model to student_score_model.pkl")
