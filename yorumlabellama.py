from textblob import TextBlob
import pandas as pd

# Veriyi yükle
df = pd.read_csv("cleaned_reddit_comments.csv")

# Eksik veya NaN değerleri temizle
df = df.dropna(subset=['cleaned_text'])

# Temizleme sonrası sadece string değerlerle çalış
df['cleaned_text'] = df['cleaned_text'].astype(str)

# Duygu analizi fonksiyonu
def sentiment_label(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

# Yorumları etiketle
df['sentiment'] = df['cleaned_text'].apply(sentiment_label)

# Sonuçları kontrol et
print(df['sentiment'].value_counts())

# Etiketlenmiş veriyi kaydet
df.to_csv("labeled_reddit_comments1.csv", index=False)
print("Etiketlenmiş veriler 'labeled_reddit_comments.csv' dosyasına kaydedildi.")
