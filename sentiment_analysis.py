import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt

# Carregar o dataset
dataset = pd.read_csv('test.csv', encoding='ISO-8859-1')

# Visualizar as primeiras linhas do dataset
print(dataset.head())

# Função para limpar tweets
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'@\S+', '', tweet)     # Remove menções
    tweet = re.sub(r'#\S+', '', tweet)     # Remove hashtags
    tweet = re.sub(r'[^A-Za-z0-9\s]', '', tweet)  # Remove caracteres especiais
    return tweet

# Aplicar limpeza aos tweets
dataset['cleaned_text'] = dataset['SentimentText'].apply(clean_tweet)

# Função para analisar sentimento
def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'positivo'
    elif analysis.sentiment.polarity == 0:
        return 'neutro'
    else:
        return 'negativo'

# Aplicar análise de sentimento
dataset['sentiment_analysis'] = dataset['cleaned_text'].apply(get_sentiment)

# Visualizar resultados
sentiments = dataset['sentiment_analysis'].value_counts()
sentiments.plot(kind='bar', color=['green', 'blue', 'red'])
plt.xlabel('Sentimentos')
plt.ylabel('Contagem')
plt.title('Distribuição de Sentimentos dos Tweets')
plt.show()


