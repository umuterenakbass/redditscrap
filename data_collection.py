import praw
import pandas as pd


reddit = praw.Reddit(
    client_id="nezJ5aiFn1Rcawqbo4Fsnw",
    client_secret="bQOfzCu1STKMjeC78s7USsUNfKmC_Q",  
    user_agent = "script:ueapplication:v1.0 (by u/uea4)"
     
)

# 'news' subreddit'inden yorumları çekin
def get_reddit_comments(subreddit_name, num_comments):
    comments = []
    subreddit = reddit.subreddit(subreddit_name)
    
    for submission in subreddit.hot(limit=50):  
        submission.comments.replace_more(limit=None)  
        for comment in submission.comments.list():
            comments.append(comment.body)
            if len(comments) >= num_comments:
                return comments
    return comments


comments = get_reddit_comments('news', 7000)




df = pd.DataFrame(comments, columns=["text"])
df.to_csv("reddit_comments1.csv", index=False)

print("4000 yorum başarıyla çekildi ve 'reddit_comments.csv' dosyasına kaydedildi.")

import re

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # URL'leri kaldır
    text = re.sub(r"@\w+", "", text)  # Kullanıcı adlarını kaldır
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Noktalama işaretlerini kaldır
    text = text.lower()  # Küçük harfe çevir
    text = text.strip()  # Başı ve sonundaki boşlukları kaldır
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
print(df.head())
