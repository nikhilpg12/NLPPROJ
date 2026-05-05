import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import classification_report, accuracy_score

test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")
eval_df = pd.read_csv("evaluation.csv")

required_columns = {"text", "author"}

for name, df in [("train", train_df), ("test", test_df), ("eval", eval_df)]:
    if not required_columns.issubset(df.columns):
        raise ValueError(f"{name}.csv must contain columns: {required_columns}")

X_test = test_df['text']
y_test = test_df['author']

X_train = train_df["text"]
y_train = train_df["author"]

X_eval = eval_df["text"]
y_eval = eval_df["author"]

word_tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    max_features=10000,
    stop_words=None
)

char_tfidf = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),
    max_features=20000
)

features = FeatureUnion([
    ("word_tfidf", word_tfidf),
    ("char_tfidf", char_tfidf)
])

model = Pipeline([
    ("features", features),
    ("clf", SGDClassifier(
        loss="log_loss",
        max_iter=8,
        tol=None,
        random_state=42
    ))
])

print("Training model (8 epochs)...")
model.fit(X_train, y_train)

print("\n=== Test Set Performance ===")
y_pred_test = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))

print("\n=== Eval Set Performance ===")
y_pred_eval = model.predict(X_eval)

print("Accuracy:", accuracy_score(y_eval, y_pred_eval))
print(classification_report(y_eval, y_pred_eval))

def predict_author(text):
    return model.predict([text])[0]

if __name__ == "__main__":
    sample_text = "Dit is 'n eenvoudige voorbeeldsin om outeurskap te toets."
    print("\nSample Prediction:", predict_author(sample_text))



