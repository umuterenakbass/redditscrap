import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd

# Veri Yükleme ve Ön İşleme
def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['cleaned_text'].astype(str).values
    labels = df['sentiment'].values
    return texts, labels

# Metinleri Sayısal Hale Getirme ve Etiketleme
def preprocess_data(texts, labels, max_words=5000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    return X, y, tokenizer, label_encoder

# LSTM Modeli Oluşturma
def create_lstm_model(input_dim, output_dim, input_length):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Veri Yükleme
file_path = 'labeled_reddit_comments1.csv'
texts, labels = load_data(file_path)

# Veri Ön İşleme
max_words = 6000
max_len = 100
X, y, tokenizer, label_encoder = preprocess_data(texts, labels, max_words, max_len)

# Eğitim ve Test Verilerini Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli Oluşturma
model = create_lstm_model(input_dim=max_words, output_dim=128, input_length=max_len)
model.summary()

# Modeli Eğitme
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=2)

# Modelin Performansını Değerlendirme
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Test Seti Tahminleri
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Detaylı Performans Metrikleri
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print(f"Weighted F1 Score: {f1:.4f}")

# Modeli Kaydetme
model.save('enhanced_lstm_model21.h5')
