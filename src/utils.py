import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple:
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    y_train['label_id'] = y_train['label'].factorize()[0]
    y_test['label_id'] = y_test['label'].factorize()[0]

    label_id_df = y_train[['label', 'label_id']].drop_duplicates().sort_values('label_id')
    label_to_id = dict(label_id_df.values)
    id_to_label = dict(label_id_df[['label_id', 'label']].values)

    features = tfidf.fit_transform(X_train.text).toarray() # Remaps the words in the train articles in the text column of 
                                                           # data frame into features (superset of words) with an importance assigned 
                                                           # based on each words frequency in the document and across documents
    labels = y_train.label_id                              # represents the category of each of the all train articles
    test_labels = y_test.label_id
    test_features = tfidf.transform(X_test.text.tolist())
    return (tfidf, features, test_features, labels, test_labels, label_to_id, id_to_label)