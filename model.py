import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.combine import SMOTEENN  # For handling imbalanced data

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("credit_rating.csv")

df.drop(columns = ['S.No', 'S.No.'], inplace = True)

missing_values = df.isnull().sum()

duplicates = df.duplicated().sum()

print(missing_values, duplicates)

if duplicates > 0:
    df.drop_duplicates(inplace=True)


categorical_columns = df.select_dtypes(include=['object']).columns
print(categorical_columns)

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])


related_cols = []
matrix = df.corr().to_dict()
for i, j in matrix.items():
  if i == 'Credit classification':
    for j, k in matrix[i].items():
      if abs(k) > 0.03:
        related_cols.append(j)
related_cols.remove('Credit classification')
print(related_cols)


X = df[related_cols]
y = df['Credit classification']



# Initialize SMOTEENN
smote_enn = SMOTEENN(random_state=42)

# Apply SMOTEENN
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Training the Logistic Regression model
logistic_model = LogisticRegression(max_iter=3000)
logistic_model.fit(X_train, y_train)

# Predicting the test set
y_pred = logistic_model.predict(X_test)

# Evaluating the model

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy: ", accuracy*100)
print("Precision: ", precision*100)
print("Recall: ", recall*100)
print("F1 Score: ", f1*100)

with open('model_new.pkl', 'wb') as file:
    pickle.dump(logistic_model, file)


print("All done")
print('------------------------------------')


