import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('wc_final_dataset.csv')

# Preprocess the data
data = data.dropna()  # Drop missing values
data['Team1'] = pd.factorize(data['Team1'])[0]
data['Team2'] = pd.factorize(data['Team2'])[0]
data['Winner'] = pd.factorize(data['Winner'])[0]

# Select features and target variable
X = data[['Team1', 'Team2', 
           'Team1 Avg Batting Ranking', 'Team2 Avg Batting Ranking', 
           'Team1 Avg Bowling Ranking', 'Team2 Avg Bowling Ranking', 
           'Team1 Total WCs participated', 'Team2 Total WCs participated', 
           'Team1 win % over Team2']]
y = data['Winner']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
