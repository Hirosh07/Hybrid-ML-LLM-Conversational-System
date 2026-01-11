import pandas as pd
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=2, random_state=42)

# Create a pipeline that combines TF-IDF vectorization and Logistic Regression

Pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
    ))
])

Pipeline.fit(x_train, y_train)
y_pred = Pipeline.predict(x_test)
print(classification_report(y_test, y_pred))

joblib.dump(Pipeline, '../../models/intend_model.pkl')
print("Model saved to ../../models/intend_model.pkl")