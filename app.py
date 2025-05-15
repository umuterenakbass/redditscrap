from flask import Flask, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Veri Yükleme
file_path = 'labeled_reddit_comments1.csv'
df = pd.read_csv(file_path)
texts = df['cleaned_text'].astype(str).values
labels = df['sentiment'].values

# Metinleri Sayısal Hale Getirme ve Etiketleme
max_words = 6000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Eğitim ve Test Verilerini Ayırma
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Dosyaları
model_files = {
    'Enhanced LSTM (50 epochs)': 'enhanced_lstm_model.h5',
    'BiLSTM (50 epochs)': 'bilstm_model.h5',
    'CNN (50 epochs)': 'cnn_model.h5',
    'Enhanced LSTM (100 epochs)': 'enhanced_lstm_model21.h5',
    'BiLSTM (100 epochs)': 'bilstm_model21.h5',
    'CNN (100 epochs)': 'cnn_model21.h5'
}

# Performans Metriklerini Hesaplama
model_metrics = {}

for model_name, model_file in model_files.items():
    # Modeli Yükleme
    model = load_model(model_file)

    # Modelin Performansını Değerlendirme
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Tahmin ve Metrik Hesaplama
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    # Sonuçları Kaydetme
    model_metrics[model_name] = {
        'accuracy': accuracy,
        'loss': loss,
        'f1_score': f1
    }

# Grafikleri Kaydetme
def save_graphs():
    metrics = ['Accuracy', 'Loss', 'F1 Score']
    for metric in metrics:
        # Anahtar adı boşlukları alt çizgi ile değiştirilmiş ve küçük harfe çevrilmiş
        metric_key = metric.replace(' ', '_').lower()
        values = [model_metrics[model][metric_key] for model in model_metrics]

        plt.figure(figsize=(10, 6))
        plt.bar(model_metrics.keys(), values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'purple', 'orange'])
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        plt.ylim(0, 1 if metric != 'Loss' else max(values) + 0.1)
        plt.xticks(rotation=15, ha='right')
        output_dir = os.path.join(os.getcwd(), 'static')
        os.makedirs(output_dir, exist_ok=True)  # static klasörünü oluştur
        output_path = os.path.join(output_dir, f'{metric_key}.png')
        plt.savefig(output_path)
        plt.close()

save_graphs()

# Ana Sayfa
@app.route('/')
def home():
    return render_template('index.html', model_metrics=model_metrics)

if __name__ == '__main__':
    app.run(debug=True)
