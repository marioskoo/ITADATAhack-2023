import re
import ast

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

# Preprocessing del testo con TF-IDF
def preprocess_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    tokens = [
        ps.stem(word.lower())
        for word in tokens
        if word.lower() not in stop_words and re.match(r"^[a-zA-Z]+$", word)
    ]
    return " ".join(tokens)

def preprocess_text_tfidf(X_train, max_len_features=1000):
    # Applica il preprocessing del testo
    X_train["text"] = X_train["text"].apply(preprocess_text)
    
    # Creazione del vettorizzatore TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=max_len_features)
    X_tfidf = tfidf_vectorizer.fit_transform(X_train["text"])
    
    # Conversione della matrice sparsa in una matrice densa
    X_tfidf_dense = X_tfidf.toarray()
    
    # Preparazione dei dati target
    y_train = X_train["labels"].tolist()
    y_train = [ast.literal_eval(item) for item in y_train]
    y_train_padded = pad_sequences(y_train, maxlen=20, padding='post', truncating='post')
    
    return X_tfidf_dense, y_train_padded

def preprocess_text_word_embedding(X_train):
    # Function to preprocess text
    def preprocess_text(text):
        ps = PorterStemmer()
        tokens = word_tokenize(text)
        tokens = [
            word
            for word in tokens
            if word not in stop_words
            and word.lower() not in months
            and re.match(r"^[a-zA-Z]+$", word)
        ]
        tokens = [ps.stem(token) for token in tokens]
        return tokens

    # Months list
    months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]

    # Applying preprocessing to X_train
    stop_words = set(stopwords.words("english"))
    X_train["text"] = X_train["text"].apply(preprocess_text)

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train["text"])
    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(X_train["text"])

    max_len_features = len(max(sequences, key=len))  # Maximum sequence length

    padded = pad_sequences(sequences, maxlen=max_len_features, padding='post', truncating='post')
    
    # Preparazione dei dati target
    y_train = X_train["labels"].tolist()
    y_train = [ast.literal_eval(item) for item in y_train]
    y_train_padded = pad_sequences(y_train, maxlen=20, padding='post', truncating='post')
    
    return padded, y_train_padded, max_len_features