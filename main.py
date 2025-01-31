import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('churn-bigml-20.csv')  # Replace with your actual file name

# Data type for each column
print(df.dtypes)

# Identify non-numeric columns
nnc = df.select_dtypes(include=['object']).columns
print("\nNon-Numeric Columns:")
print(nnc)

# Encode non-numeric columns
encoder = LabelEncoder()
for col in ['State', 'International plan', 'Voice mail plan']:
    df[col] = encoder.fit_transform(df[col])

# Verify the transformation
print(df.head())

# Convert 'Churn' to numeric
df['Churn'] = df['Churn'].astype(int)

# Verify the conversion
print(df['Churn'].head())

# Define features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Testing target shape: {y_test.shape}")

# Define the parameter grid for RandomizedSearchCV
# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': np.arange(100, 1001, 100),  # Try from 100 to 1000 trees
    'max_depth': [None, 10, 20, 30, 40, 50],  # No depth limit or limited depth
    'min_samples_split': [2, 5, 10],  # Number of samples to split nodes
    'min_samples_leaf': [1, 2, 4],  # Number of samples at leaf node
    'max_features': ['sqrt', 'log2'],  # Valid options
    'bootstrap': [True, False]  # Bootstrap samples or not
}


# Initialize the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Perform RandomizedSearchCV to find the best hyperparameters
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                                   n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the model to the training data
random_search.fit(X_train, y_train)

# Get the best model with optimized hyperparameters
best_rf = random_search.best_estimator_

# Predict churn on the test set
y_pred_best_rf = best_rf.predict(X_test)

# Evaluate accuracy
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
print(f"Accuracy after tuning: {accuracy_best_rf:.2f}")

# Print the confusion matrix and classification report
print("Confusion Matrix after tuning:")
print(confusion_matrix(y_test, y_pred_best_rf))

print("Classification Report after tuning:")
print(classification_report(y_test, y_pred_best_rf))

# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Check new class distribution after resampling
print("Original class distribution:", y_train.value_counts().to_dict())
print("Resampled class distribution:", pd.Series(y_resampled).value_counts().to_dict())

# Fit the tuned Random Forest model with resampled data
best_rf.fit(X_resampled, y_resampled)

# Predict churn on the test set after SMOTE
y_pred_resampled_rf = best_rf.predict(X_test)

# Evaluate accuracy after SMOTE and model tuning
accuracy_resampled_rf = accuracy_score(y_test, y_pred_resampled_rf)
print(f"Accuracy after SMOTE & Tuning: {accuracy_resampled_rf:.2f}")

# Print confusion matrix and classification report after resampling
print("Confusion Matrix after SMOTE & Tuning:")
print(confusion_matrix(y_test, y_pred_resampled_rf))

print("Classification Report after SMOTE & Tuning:")
print(classification_report(y_test, y_pred_resampled_rf))
