#!/usr/bin/env python
# coding: utf-8

# # Model Word Embeddings
# Model Word Embeedings (osadzanie słów) jest techniką wykorzystywaną do reprezentowania tekstu w przetwarzaniu języka naturalnego. W przeciwieństwie do prostszych metod, takich jak bag of words, ma na celu uchwycenie znaczenia słów i kontekstowych relacji między nimi, a nie tylko określenia częstotliwości ich występowania.
# 
# Cechy Modelu Word Embeedings:
# 1) Poszczególne słowa są reprezentowane jako wektory o wartościach rzeczywistych w określonej przestrzeni wektorowej. Każde słowo jest mapowane na jeden wektor. <br>
# 2) Podobne słowa znajdują się bliżej siebie (podobne jeżeli chodzi o ich znaczenie, sposób użycia w języku i kontekst) . <br>
# 3) Każdy wymiar wektora zawiera określone informacje o znaczeniu lub kontekście słowa. Umożliwia to uchwycenie złożonych relacji między słowami. <br>
# 
# - Tymczasem w prostszych modelach każde słowo jest reprezentowane jako osobna cecha, a obecność lub częstotliwość słów jest wykorzystywana do budowy macierzy. Różne wyrazy mają różne reprezentacje, niezależnie od tego, jak są używane.
# 
# ## Algorytmy osadzania słów
# Modele Word Embeedings są zwykle wstępnie przeszkolone przy użyciu metod uczenia bez nadzoru na dużych zbiorach danych. Najpopularniejszymi takimi metodami są: Embedding Layer, Word2Vec, GloVe i FastText. Modele te są trenowane na dużych korpusach tekstowych, co pozwala im uchwycić ogólne relacje semantyczne. <br>
# 
# 1) Embedding Layer: w modelach głębokiego uczenia się mapuje dyskretne słowa lub tokeny na ciągłe wektory, umożliwiając reprezentowanie ich jako gęstych osadzeń w sieci neuronowej. <br>
# 2) Word2Vec: uczy się osadzania słów poprzez przewidywanie wyrazów sąsiadujących wobec słowa docelowego oraz poprzez przewidywanie szerszego kontekstu. W ten sposób wychwytuję relacje semantyczne i podobieństwa. <br>
# 3) GloVe: generuje osadzenia słów, wykorzystując globalne statystyki ich współwystępowania. <br>
# 4) FastText: jest rozszerzeniem Word2Vec, które wprowadza informacje o pod-słowach poprzez reprezentowanie każdego wyrazu jako worka n-gramów znaków, co pozwala na obsługę słów spoza słownika.
# 
# ## Opracowanie modelu
# Opracowanie modelu na potrzeby tego projektu przebiegało według następujących kroków:
# 1) Ładowanie potrzebnych bibliotek i zbiorów danych. <br> 
# 2) Połączenie i przetasowanie zestawów danych. <br>
# 3) Czyszczenie danych tekstowych poprzez usunięcie niepotrzebnych znaków, cyfr, znaków interpunkcyjnych, wszelkich symboli specjalnych oraz konwersje tekstu na małe litery. <br> 
# 4) Tokenizacja: podział oczyszczonego tekstu na pojedyncze słowa lub tokeny. <br> 
# 5) Usuwanie 'Stopwords' - powszechnie używanych słów, które często pojawiają się w języku, ale nie wnoszą wiele do ogólnego zrozumienia tekstu (np. "the", "is", "and", "a", "an"). <br> 
# 6) Stemming: zredukowanie słów do ich formy podstawowej lub źródłowej, znanej jako "rdzeń". <br> 
# 7) Dzielenie danych: podział zbioru danych na zestawy treningowe i testowe. Jeden zostanie wykorzystany do wytrenowania modelu, a drugi do oceny jego wydajności. <br> 
# 8) Tworzenie osadzeń słów za pomocą algorytmu Word2Vec. <br>
# 9) Konwersja danych tekstowych na osadzenia słów. <br> 
# 10) Trenowanie modelu klasyfikacji (Model Support Vector Machines (SVM)). <br> 
# 11) Ocena Modelu. <br>
# 12) Załadowanie i przygotowanie nowych danych. <br> 
# 13) Wykorzystanie modelu do tworzenia predykcji na podstawie nowych danych. <br>
# 14) Ocena dokładności predykcji.
# 
# ## Na czym polega ten projekt?
# Celem projektu jest stworzenie modelu, który przewidywałby, czy artykuł jest fake newsem, czy nie, na podstawie jego tytułu. Źródłem danych są dwa zbiory - jeden zawiera wyłącznie prawdziwe artykuły, a drugi wyłącznie fałszywe. Każdy z zestawów zawiera ponad 20 000 rekordów, ale tylko cztery tysiące zostały wykorzystane w projekcie (pierwszy tysiąc z każdego zestawu do uczenia algorytmu i testowania oraz ostatni tysiąc z każdego do tworzenia nowych prognoz)
# 
# Źródło danych: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

# # 1. Ładowanie potrzebnych bibliotek i zbiorów danych

# In[1]:


import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
import numpy as np
import regex as re

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

from gensim.models import Word2Vec


True_News = pd.read_csv("True.csv",sep=",", nrows=1000) #Pobieram pierwszy tysiąc rekordów ze zbioru z prawdziwymi artykułami
Fake_News = pd.read_csv("Fake.csv",sep=",", nrows=1000) #Pobieram pierwszy tysiąc rekordów ze zbioru z fałszywymi artykułami


# # 2. Połączenie i przetasowanie zestawów danych:
# 

# In[2]:


#Dodaję kolumny z etykietami
Fake_News['label'] = 'fake'
True_News['label'] = 'true'

#Biorę tylko kolumnę z tytułami i kolumnę z etykietami
True_Text = True_News[['title','label']]
Fake_Text = Fake_News[['title','label']]

combined_df = pd.concat([Fake_Text, True_Text], ignore_index=True) #łączę obydwa zbiory
shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True) #tasuję połączony zbiór


# In[25]:


shuffled_df.head(10)
#W ten sposób otrzymuję zbiór, w którym zmieszane są ze sobą tytuły prawdziwych i fałszywych artykułów (po 1000 każdego typu)


# # (3, 4, 5, 6). Czyszczenie tekstu, tokenizacja, usuwanie 'stopwords', stemming

# In[4]:


def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text) #usuwanie znaków niealfabetycznych
    cleaned_text = cleaned_text.lower()
    return cleaned_text

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    filtered_tokens = [token for token in tokens if not token in stop_words]
    return filtered_tokens

stemmer = PorterStemmer()

def apply_stemming(tokens):
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

shuffled_df['title'] = shuffled_df['title'].apply(clean_text)
shuffled_df['title'] = shuffled_df['title'].apply(tokenize_text)
shuffled_df['title'] = shuffled_df['title'].apply(remove_stopwords)
shuffled_df['title'] = shuffled_df['title'].apply(apply_stemming)


# In[26]:


print(shuffled_df.head(10)) #Tekst tytułów po przygotowaniu wraz z etykietami


# # 7. Podział zbioru danych na zestawy treningowe i testowe

# In[7]:


#podzielenie danych na zbiór treningowy(80%) i zbiór testowy (20%)
X_train, X_test, y_train, y_test = train_test_split(shuffled_df['title'], shuffled_df['label'], test_size=0.2, random_state=42) 


# # 8. Tworzenie osadzeń słów za pomocą alogrytmu Word2Vec

# In[8]:


word2vec_model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1, sg=1)
vocab_size = len(word2vec_model.wv.key_to_index)


# # 9. Konwersja danych tekstowych na osadzenia słów
# Definiuję funkcję get_word_embeddings, która konwertuje każdy tokenizowany tekst na osadzenia słów. Jeśli słowo jest obecne w słowniku Word2Vec, jego osadzenie jest dodawane do listy. Jeśli słowo nie występuje, zamiast niego dodawany jest wektor zerowy. Na koniec funkcja oblicza średnią z osadzeń na liście. 
# 
# Stosuję tę funkcję wobec danych tekstowych w zestawach treningowych i tekstowych. 

# In[9]:


def get_word_embeddings(text):
    embeddings = []
    for word in text:
        if word in word2vec_model.wv:
            embeddings.append(word2vec_model.wv[word])
    if len(embeddings) == 0:
        embeddings.append(np.zeros(word2vec_model.vector_size))
    return np.mean(embeddings, axis=0)

X_train_embeddings = np.array([get_word_embeddings(text) for text in X_train])
X_test_embeddings = np.array([get_word_embeddings(text) for text in X_test])

# Wyniki osadzenia
print(X_train_embeddings)
print(X_test_embeddings)


# # 10. Trenowanie modelu klasyfikacji (Model Support Vector Machines (SVM))

# In[10]:


classifier = SVC()
classifier.fit(X_train_embeddings, y_train)


# # 11. Ocena Modelu

# In[11]:


accuracy = classifier.score(X_test_embeddings, y_test)
print("Accuracy:", accuracy)
#60% dokładności na zbiorze testowym 


# # 12. Załadowanie i przygotowanie nowych danych

# In[29]:


True_News_New = pd.read_csv("True.csv",sep=",", skiprows=lambda x: x != 0 and x < (1000 - 1), nrows=1000)
Fake_News_New = pd.read_csv("Fake.csv",sep=",", skiprows=lambda x: x != 0 and x < (1000 - 1), nrows=1000)

Fake_News_New['label'] = 'fake'
True_News_New['label'] = 'true'

True_Text_New = True_News_New[['title','label']]
Fake_Text_New = Fake_News_New[['title','label']]

combined_df_New = pd.concat([Fake_Text_New, True_Text_New], ignore_index=True)
shuffled_df_New = combined_df_New.sample(frac=1, random_state=42).reset_index(drop=True)


# In[30]:


shuffled_df_New.head(10) #nowe dane przed przygotowaniem


# In[31]:


shuffled_df_New['title'] = shuffled_df_New['title'].apply(clean_text)
shuffled_df_New['title'] = shuffled_df_New['title'].apply(tokenize_text)
shuffled_df_New['title'] = shuffled_df_New['title'].apply(remove_stopwords)
shuffled_df_New['title'] = shuffled_df_New['title'].apply(apply_stemming)

new_data = shuffled_df_New['title']
actual_labels = shuffled_df_New['label']

shuffled_df_New.head(10) #nowe dane po przygotowaniu


# # 13. Wykorzystanie modelu do tworzenia predykcji na podstawie nowych danych.

# In[38]:


new_data_embeddings = [get_word_embeddings(text) for text in new_data ]
predictions = classifier.predict(new_data_embeddings)


# In[42]:


# for data, prediction in zip(new_data , predictions):
#     print(f"Data: {data}")
#     print(f"Prediction: {prediction}")
#     print()

#Predykcje dotyczące poszczególnych tytułów


# # 14. Ocena dokładności predykcji

# In[40]:


results_df = pd.DataFrame({'Data': new_data, 'Prediction': predictions, 'Actual': actual_labels})
results_df.head(10) #Porównanie predykcji ('Prediction') z rzeczywistymi etykietami ('Actual') dla poszczególnych artykułów 


# In[41]:


accuracy = accuracy_score(actual_labels, predictions)
print("Accuracy:", accuracy)

#56% dokładności na mieszanym zbiorze prawdziwych i fałszywych artykułów (w porównaniu do 60% na zbiorze testowym)


# In[ ]:




