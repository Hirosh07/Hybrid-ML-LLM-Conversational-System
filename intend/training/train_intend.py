import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df=pd.read_csv('../../data/intend.csv')
# Split the data into features and labels
x = df["text"]
y = df["intent"]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y, random_state=42)

# Create a pipeline that combines TF-IDF vectorization and Logistic Regression

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=1,
        max_features=12000
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
    ))
])

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

os.makedirs('../../models', exist_ok=True)
joblib.dump(model, '../../models/intend_x_train.pkl')
print("Model saved to ../../models/intend_model.pkl")