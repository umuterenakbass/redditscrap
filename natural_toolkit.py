import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys  # exit() fonksiyonu için sys modülünü içe aktarın

# VADER duygu analizini başlat
analyzer = SentimentIntensityAnalyzer()

# CSV dosyasından Reddit yorumlarını yükle
try:
    df = pd.read_csv("reddit_comments.csv", on_bad_lines='skip', encoding='utf-8')
except Exception as e:
    print(f"CSV dosyası yüklenirken hata oluştu: {e}")
    sys.exit()  # Hata durumunda programı sonlandır

# NaN (eksik) değerleri temizle
df = df.dropna(subset=['text'])

# Her yorumun duygu puanını hesaplayın
def analyze_sentiment(comment):
    try:
        score = analyzer.polarity_scores(comment)
        if score['compound'] >= 0.05:
            return "positive"
        elif score['compound'] <= -0.05:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return "neutral"  # Hatalı durumlarda nötr olarak etiketleyin

# Yorumlara duygu etiketi ekleyin
df['sentiment'] = df['text'].apply(analyze_sentiment)

# Sonuçları kaydedin
try:
    df.to_csv("labeled_reddit_comments.csv", index=False)
    print("Yorumlar başarıyla etiketlendi ve 'labeled_reddit_comments.csv' dosyasına kaydedildi.")
except Exception as e:
    print(f"Sonuçlar kaydedilirken hata oluştu: {e}")
