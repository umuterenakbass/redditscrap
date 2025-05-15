import pandas as pd
import re

# Temizleme fonksiyonu
def clean_text(text):
    text = re.sub(r"http\S+|www.\S+", "", text)  # URL'leri kaldır
    text = re.sub(r"@\w+", "", text)  # Mention'ları kaldır
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Noktalama işaretlerini kaldır
    text = re.sub(r"\s+", " ", text).strip()  # Fazla boşlukları temizle
    text = text.lower()  # Küçük harfe çevir
    return text

# Orijinal veriyi yükle
df = pd.read_csv("reddit_comments1.csv")

# Temizleme işlemi
df['cleaned_text'] = df['text'].apply(clean_text)

# Uzunluk sütunu ekle
df['length'] = df['cleaned_text'].apply(len)

# Temizlenmiş veriyi kaydet
df[['cleaned_text']].to_csv("cleaned_reddit_comments.csv", index=False)

df['cleaned_text'] = df['cleaned_text'].apply(lambda x: x[:1000] if len(x) >= 0 else x)
df['length'] = df['cleaned_text'].apply(len)  # Uzunluk sütununu güncelleyin


# İstatistiksel özet
statistics = df['length'].describe()
print(statistics)

df = df[df['length'] > 0]
print(f"Kalan yorum sayısı: {len(df)}")


