import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    # CSV dosyasını yükle
    df = pd.read_csv(file_path)
    print("Veri sütunları:", df.columns)  # Sütun isimlerini kontrol et
    return df

def preprocess_data(df):
    # Veri setinde bir 'sentiment' sütunu olup olmadığını kontrol et
    if 'sentiment' not in df.columns:
        print("Veri setinde 'sentiment' sütunu bulunamadı. Etiketlerinizi manuel olarak eklemeniz gerekebilir.")
        raise KeyError("Veri setinde 'sentiment' sütunu bulunamadı.")
    else:
        # 'LabelEncoder' ile 'sentiment' etiketlerini sayısal değerlere dönüştür
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['sentiment'])
        print("Label Encoding yapıldı: ", dict(zip(le.classes_, le.transform(le.classes_))))
    return df

def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# Örnek kullanım:
if __name__ == "__main__":
    file_path = 'labeled_reddit_comments.csv'
    df = load_data(file_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
