import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import nltk
from nltk.corpus import stopwords
import re

# Carregar o conjunto de treinamento
train_df = pd.read_csv('./data/amazon_review_polarity_csv/train.csv', header=None, names=['polarity', 'title', 'text'])

# Carregar o conjunto de teste
test_df = pd.read_csv('./data/amazon_review_polarity_csv/test.csv', header=None, names=['polarity', 'title', 'text'])

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove caracteres especiais e números
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Transforma em minúsculas
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Aplica o pré-processamento nos textos
train_df['text_clean'] = train_df['text'].apply(preprocess_text)
test_df['text_clean'] = test_df['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['text_clean'])
X_test = vectorizer.transform(test_df['text_clean'])

# As labels
y_train = train_df['polarity']
y_test = test_df['polarity']

# Inicializa e treina o modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Faz as previsões
y_pred = model.predict(X_test)

# Avalia a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

# Relatório de classificação
print(classification_report(y_test, y_pred))

# Matriz de confusão
print(confusion_matrix(y_test, y_pred))

# Selecionando aleatoriamente 10 frases do conjunto de teste
sampled_df = test_df.sample(n=10, random_state=42)

# Prevê os resultados para essas frases
sampled_df['predicted_polarity'] = model.predict(X_test[sampled_df.index])

# Selecionando as colunas desejadas para o CSV
result_df = sampled_df[['text', 'polarity', 'predicted_polarity']]

# Salvando os resultados
result_df.to_csv('resultados.csv', index=False)

print("Arquivo 'resultados.csv' criado com 10 amostras de frases do conjunto de teste.")

scores = cross_val_score(model, X_train, y_train, cv=5)  # fazendo 5-fold cross validation
print("Cross-validation scores:", scores)
