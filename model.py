import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("credit_rating.csv")

# Encode categorical features
label_encoder = LabelEncoder()
df_encoded = df.apply(label_encoder.fit_transform)

# Split data into features and target
X = df_encoded.drop(['S.No', 'S.No.', 'Credit classification'], axis=1)
y = df_encoded['Credit classification']

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
lor_model = LogisticRegression(max_iter=10000)
lor_model.fit(x_train, y_train)
lor_score = lor_model.score(x_test, y_test)
print(f"Logistic Regression score: {lor_score}")

# Evaluate logistic regression model
y_pred = lor_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Logistic Regression accuracy: {accuracy}")
print(f"Logistic Regression F1 score: {f1}")

print('------------------------------------')

# Train Decision Tree model
dtc_model = DecisionTreeClassifier()
dtc_model.fit(x_train, y_train)
dtc_score = dtc_model.score(x_test, y_test)
print(f"Decision Tree score: {dtc_score}")

# Evaluate Decision Tree model
y_pred = dtc_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Decision Tree accuracy: {accuracy}")
print(f"Decision Tree F1 score: {f1}")

# Serialize the ML model
with open('model.pkl', 'wb') as file:
    pickle.dump(lor_model, file)

# testing the data
# columns = ['CHK_ACCT','Duration','History','Purpose of credit','Credit Amount',
#         'Balance in Savings A/C','Employment','Install_rate','Marital status',
#         'Co-applicant','Present Resident','Real Estate','Age','Other installment',
#         'Residence','Num_Credits','Job','No. dependents','Phone','Foreign']
# sample = [[0, 6, 2, 6, 1169, 4, 2, 4, 3, 2, 4, 3, 67, 1, 1, 2, 1, 1, 0, 1]]
# new_df = pd.DataFrame(sample, columns=columns)
# prediction = list(lor_model.predict(new_df))
# print(prediction)

