# 1. Define Input Variables // Create Synthetic Dataset
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 200

# Generate synthetic network metrics
data = pd.DataFrame({
    'packet_loss': np.random.uniform(0, 10, num_samples),    # % of packet loss
    'latency': np.random.uniform(10, 300, num_samples),      # ms
    'throughput': np.random.uniform(10, 100, num_samples),   # Mbps
    'cpu_usage': np.random.uniform(20, 100, num_samples),    # % CPU usage
    'memory_usage': np.random.uniform(20, 100, num_samples)  # % memory usage
})

##### 
data['failure'] = ((data['packet_loss'] > 5) | 
                   (data['latency'] > 200) | 
                   (data['cpu_usage'] > 85)).astype(int)

# Save dataset to CSV 
data.to_csv('network_data.csv', index=False)



# 2 Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Separate input features (X) and target label (y)
X = data[['packet_loss', 'latency', 'throughput', 'cpu_usage', 'memory_usage']]
y = data['failure']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Show the size of each set
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)



# 3 Train the Logistic Regression model

# Create the Logistic Regression model
model = LogisticRegression()

# Train the model on training data
model.fit(X_train, y_train)

# Model is now trained and has learned patterns between features and failure
print("Model training completed.")


# Show which features are most influential
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.3f}")


# 4 Make predictions on the test data
# Predict labels (0 or 1) for X_test
y_pred = model.predict(X_test)

# Optional: Predict probabilities of failure (between 0 and 1)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of failure = 1

# Show first 10 predictions vs actual labels
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Failure_Probability': y_prob.round(2)
})
print(comparison.head(10))


import joblib

# Save the trained model to a file
joblib.dump(model, 'network_model.pkl')
print("Model saved to network_model.pkl")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal','Failure'], yticklabels=['Normal','Failure'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Determine which predictions were correct
correct = (y_test.values == y_pred)
wrong = ~correct

plt.figure(figsize=(12,4))


plt.figure(figsize=(12,4))
plt.step(range(len(y_test)), y_test.values, where='mid', label='Actual Failure', color='black')
plt.scatter(range(len(y_test)), y_pred, color='blue', label='Predicted Failure', marker='o')
plt.xlabel('Sample Index')
plt.ylabel('Failure (0=Normal, 1=Failure)')
plt.title('Predicted vs Actual Failures — Step Diagram')
plt.yticks([0,1], ['Normal','Failure'])
plt.legend()
plt.grid(True)
plt.show()
