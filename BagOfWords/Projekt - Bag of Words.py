#!/usr/bin/env python
# coding: utf-8

# # Model Bag of Words
# Model bag-of-words (worek słów) jest popularnym i prostym podejściem do reprezentowania danych tekstowych w przetwarzaniu języka naturalnego. Traktuje on fragment tekstu jako nieuporządkowany zbiór lub "worek" pojedynczych słów, pomijając gramatykę, kolejność słów i kontekst. Model ten reprezentuje tekst poprzez utworzenie słownika unikalnych słów, a następnie ilościowe określenie ich występowania w danym dokumencie. 
# 
# Model bag-of-words jest potrzebny do reprezentowania tekstu, ponieważ zapewnia skuteczny sposób konwersji nieustrukturyzowanych danych tekstowych na ustrukturyzowaną reprezentację numeryczną, z którą mogą sobie poradzić algorytmy uczenia maszynowego.
# 
# ## Opracowanie modelu
# Opracowanie modelu na potrzeby tego projektu przebiegało według następujących kroków:
# 1) Ładowanie potrzebnych bibliotek i zbiorów danych. <br> 
# 2) Czyszczenie danych tekstowych poprzez usunięcie niepotrzebnych znaków, cyfr, znaków interpunkcyjnych, wszelkich symboli specjalnych oraz konwersje tekstu na małe litery. <br> 
# 3) Tokenizacja: podział oczyszczonego tekstu na pojedyncze słowa lub tokeny. <br> 
# 4) Usuwanie 'Stopwords' - powszechnie używanych słów, które często pojawiają się w języku, ale nie wnoszą wiele do ogólnego zrozumienia tekstu (np. "the", "is", "and", "a", "an"). <br> 
# 5) Stemming: zredukowanie słów do ich formy podstawowej lub źródłowej, znanej jako "rdzeń". <br> 
# 6) Tworzenie słownika: stworzenie zestawu unikalnych słów poprzez zebranie wszystkich tokenów ze zbiorów danych. <br> 
# 7) Wektoryzacja: konwersja danych tekstowych na numeryczne wektory, które mogą być wykorzystane jako dane wejściowe do modelu uczenia maszynowego. W tym celu wypróbowany został CountVectorizer, TfidfVectorizer oraz HashingVectorizer - trzy schematy, które udostępnia biblioteka scikit-learn do budowy modelu Bag of Words. <br> 
# 8) Dzielenie danych: podział zbioru danych na zestawy treningowe i testowe. Jeden zostanie wykorzystany do wytrenowania modelu, a drugi do oceny jego wydajności. <br> 
# 9) Trening modelu: użycie metody regresji logistycznej. Regresja logistyczna jest algorytmem uczenia nadzorowanego, który może być używany do klasyfikowania dokumentów tekstowych na podstawie ich cech. <br> 
# 10) Ocena modelu: wykorzystanie testowego zbioru danych do oceny wydajności wyszkolonego modelu. <br> 
# 11) Załadowanie i przygotowanie nowych danych. <br> 
# 12) Wykorzystanie modelu do tworzenia predykcji na podstawie nowych danych.  
# 13) Ocena dokładności predykcji. <br> 
# 
# ## Na czym polega ten projekt?
# Celem projektu jest stworzenie modelu, który przewidywałby, czy artykuł jest fake newsem, czy nie, na podstawie jego tytułu. Źródłem danych są dwa zbiory - jeden zawiera wyłącznie prawdziwe artykuły, a drugi wyłącznie fałszywe. Każdy z zestawów zawiera ponad 20 000 rekordów, ale tylko cztery tysiące zostały wykorzystane w projekcie (pierwszy tysiąc z każdego zestawu do uczenia algorytmu i testowania oraz ostatni tysiąc z każdego do tworzenia nowych prognoz),
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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression


True_News = pd.read_csv("True.csv",sep=",", nrows=1000) #Pobieram pierwszy tysiąc rekordów ze zbioru z prawdziwymi artykułami
True_Text = True_News['title'] #Biorę tylko kolumnę z tytułami 

Fake_News = pd.read_csv("Fake.csv",sep=",", nrows=1000) #Pobieram pierwszy tysiąc rekordów ze zbioru z fałszywymi artykułami
Fake_Text = Fake_News['title'] #Biorę tylko kolumnę z tytułami 


# In[50]:


True_News.head(10) #Tak wygląda początkowy zbiór danych (prawdziwe artykuły)


# In[45]:


Fake_News.head(10) #Tak wygląda początkowy zbiór danych (fałszywe artykuły)


# In[46]:


True_Text.head(10) #Tak wygląda sam tekst tytułów (prawdziwe artykuły)


# In[47]:


Fake_Text.head(10) #Tak wygląda sam tekst tytułów (fałszywe artykuły)


# # (2, 3, 4, 5). Czyszczenie tekstu, tokenizacja, usuwanie 'stopwords', stemming 

# In[6]:


def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text) #usuwanie znaków niealfabetycznych
    cleaned_text = cleaned_text.lower()
    return cleaned_text
    
True_Text1 = True_Text.apply(clean_text)
Fake_Text1 = Fake_Text.apply(clean_text)

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

True_Text2 = True_Text1.apply(tokenize_text)
Fake_Text2 = Fake_Text1.apply(tokenize_text)

stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    filtered_tokens = [token for token in tokens if not token in stop_words]
    return filtered_tokens
    
True_Text3 = True_Text2.apply(remove_stopwords)
Fake_Text3 = Fake_Text2.apply(remove_stopwords)

stemmer = PorterStemmer()

def apply_stemming(tokens):
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

True_Text4 = True_Text3.apply(apply_stemming)
Fake_Text4 = Fake_Text3.apply(apply_stemming)


# In[48]:


True_Text4.head(10) #Tekst tytułów po przygotowaniu (prawdziwe)


# In[49]:


Fake_Text4.head(10) #Tekst tytułów po przygotowaniu (fałszywe)


# # 6. Tworzenie słownika

# In[51]:


all_filtered_tokens = list(Fake_Text4) + list(True_Text4)
#print(all_filtered_tokens)


# # 7. Wektoryzacja
# ## CountVectorizer (liczba wystąpień danego słowa w tekście)

# In[13]:


preprocessed_texts = [' '.join(tokens) for tokens in all_filtered_tokens]


vectorizer = CountVectorizer() 
vectorizer.fit(preprocessed_texts)

X_fake = vectorizer.transform([' '.join(tokens) for tokens in Fake_Text4])
X_true = vectorizer.transform([' '.join(tokens) for tokens in True_Text4])

X_fake_df = pd.DataFrame(X_fake.toarray(), columns=vectorizer.get_feature_names())
X_true_df = pd.DataFrame(X_true.toarray(), columns=vectorizer.get_feature_names())


# In[14]:


print(X_true_df.head(10)) #Liczba wystąpień danego słowa w tekście (prawdziwe)


# In[15]:


print(X_fake_df.head(10)) #Liczba wystąpień danego słowa w tekście (fałszywe)


# In[54]:


#print(vectorizer.vocabulary_) #indeksy przydzielone danym słowom


# ## TfidfVectorizer (obliczanie częstotliwości słów)

# In[18]:


vectorizer = TfidfVectorizer()
vectorizer.fit(preprocessed_texts)

X_fake = vectorizer.transform([' '.join(tokens) for tokens in Fake_Text4])
X_true = vectorizer.transform([' '.join(tokens) for tokens in True_Text4])

X_fake_df = pd.DataFrame(X_fake.toarray(), columns=vectorizer.get_feature_names())
X_true_df = pd.DataFrame(X_true.toarray(), columns=vectorizer.get_feature_names())


# In[19]:


print(X_true_df.head(10)) #Częstotliwość występowania danego słowa (prawdziwe)


# In[20]:


print(X_fake_df.head(10)) #Częstotliwość występowania danego słowa (fałszywe)


# ## HashingVectorizer (mapowanie każdego słowa do określonego indeksu w wektorze o stałej długości za pomocą funkcji haszującej.)

# In[21]:


vectorizer = HashingVectorizer(n_features=50)
X_fake = vectorizer.transform([' '.join(tokens) for tokens in Fake_Text4])
X_true = vectorizer.transform([' '.join(tokens) for tokens in True_Text4])

X_fake_df = pd.DataFrame(X_fake.toarray())
X_true_df = pd.DataFrame(X_true.toarray())


# In[22]:


print(X_fake_df.head(10))


# In[23]:


print(X_true_df.head(10))


# In[24]:


# Największą dokładność okazał się mieć model bazujący na wektoryzacji za pomocą CountVectorizer
vectorizer = CountVectorizer() 
vectorizer.fit(preprocessed_texts)

X_fake = vectorizer.transform([' '.join(tokens) for tokens in Fake_Text4])
X_true = vectorizer.transform([' '.join(tokens) for tokens in True_Text4])

X_fake_df = pd.DataFrame(X_fake.toarray(), columns=vectorizer.get_feature_names())
X_true_df = pd.DataFrame(X_true.toarray(), columns=vectorizer.get_feature_names())


# In[52]:


X_fake_df.head(10) #Liczba wystąpień danego słowa w tekście (fałszywe)


# In[53]:


X_true_df.head(10) #Liczba wystąpień danego słowa w tekście (prawdziwe)


# # 8. Dzielenie danych

# In[28]:


X = pd.concat([X_fake_df, X_true_df], axis=0) #łączenie obydwu ramek danych 

#stworzenie dwóch serii - jedna z fałszywymi artykułami i jedna z prawdziwymi
y_fake = pd.Series([1] * len(X_fake_df)) #1 jako etykieta fałszywych
y_true = pd.Series([0] * len(X_true_df)) #0 jako etykieta prawdziwych
y = pd.concat([y_fake, y_true], axis=0) #połączenie dwóch serii 

#podzielenie danych na zbiór treningowy(80%) i zbiór testowy (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# # 9. Trening modelu

# In[29]:


model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred


# # 10. Ocena Modelu

# In[30]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

recall = recall_score(y_test, y_pred)
print("Recall:", recall)

f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)

#92% dokładności na zbiorze testowym 


# # 11. Załadowanie i przygotowanie nowych danych

# In[31]:


True_News_New = pd.read_csv("True.csv",sep=",", skiprows=lambda x: x != 0 and x < (1000 - 1), nrows=1000) #ostatnie 1000 rekordów
True_Text_New = True_News_New['title']

Fake_News_New = pd.read_csv("Fake.csv",sep=",", skiprows=lambda x: x != 0 and x < (1000 - 1), nrows=1000) #ostatnie 1000 rekordów
Fake_Text_New = Fake_News_New['title']


# In[32]:


True_Text_New1 = True_Text_New.apply(clean_text)
Fake_Text_New1 = Fake_Text_New.apply(clean_text)

True_Text_New2 = True_Text_New1.apply(tokenize_text)
Fake_Text_New2 = Fake_Text_New1.apply(tokenize_text)

True_Text_New3 = True_Text_New2.apply(remove_stopwords)
Fake_Text_New3 = Fake_Text_New2.apply(remove_stopwords)

True_Text_New4 = True_Text_New3.apply(apply_stemming)
Fake_Text_New4 = Fake_Text_New3.apply(apply_stemming)

X_fake_New = vectorizer.transform([' '.join(tokens) for tokens in Fake_Text_New4])
X_true_New = vectorizer.transform([' '.join(tokens) for tokens in True_Text_New4])

X_fake_New_df = pd.DataFrame(X_fake_New.toarray(), columns=vectorizer.get_feature_names())
X_true_New_df = pd.DataFrame(X_true_New.toarray(), columns=vectorizer.get_feature_names())


# # 12. Wykorzystanie modelu do tworzenia predykcji na podstawie nowych danych

# ### Case_1 - Do modelu wprowadzamy zestaw z samymi fałszywymi artykułami i tworzymy predykcje 

# In[33]:


new_predictions1 = model.predict(X_fake_New_df)

labels = ['true', 'fake']
new_labels = [labels[prediction] for prediction in new_predictions1]
X_fake_New_df['predicted_label'] = new_labels


# In[55]:


results_df = pd.DataFrame({'Data': Fake_Text_New4 , 'Prediction' : new_labels})
results_df.head(10) #Predykcje dotyczące poszczególnych tytułów


# In[35]:


new_predictions_1 = model.predict(X_fake_New)
print(new_predictions_1) #W tym ujęciu widzimy, że zdecydowana większość artykułów została zakwalifikowana jako fałszywe (0 - prawdziwe, 1 - fałszywe)


# ### Case_2 - Do modelu wprowadzamy zestaw z samymi prawdziwymi artykułami i tworzymy predykcje 

# In[36]:


new_predictions2 = model.predict(X_true_New_df)

labels = ['true', 'fake']
new_labels = [labels[prediction] for prediction in new_predictions2]
X_true_New_df['predicted_label'] = new_labels


# In[56]:


results_df = pd.DataFrame({'Data': True_Text_New4 , 'Prediction' : new_labels})
results_df.head(10) #Predykcje dotyczące poszczególnych tytułów


# In[39]:


new_predictions_2 = model.predict(X_true_New)
print(new_predictions2) #W tym ujęciu widzimy, że zdecydowana większość artykułów została zakwalifikowana jako prawdziwe (0 - prawdziwe, 1 - fałszywe)


# # 13. Ocena dokładności predykcji

# ### Case_1 (fałszywe artykuły)

# In[57]:


X_fake_New_df['Actual_Label'] = "fake"
Actual_Label_Fake = X_fake_New_df['Actual_Label']
Predicted_Label_Fake = X_fake_New_df['predicted_label']

results_df = pd.DataFrame({'Data': Fake_Text_New4 , 'Prediction' : Predicted_Label_Fake, 'Actual': Actual_Label_Fake})
results_df.head(10) #Porównanie predykcji ('Prediction') z rzeczywistymi etykietami ('Actual') dla poszczególnych artykułów 


# In[41]:


X_fake_New_df['Actual_Label'] = 1
Actual_Label_Fake = X_fake_New_df['Actual_Label']

accuracy_New = accuracy_score(Actual_Label_Fake, new_predictions1 )
print("Accuracy:", accuracy_New)

#90% dokładności na zbiorze fałszywych artykułów (w porównaniu do 92% na zbiorze testowym)


# ### Case_2 (prawdziwe artykuły)

# In[58]:


X_true_New_df['Actual_Label'] = "true"
Actual_Label_True = X_true_New_df['Actual_Label']
Predicted_Label_True = X_true_New_df['predicted_label']

results_df = pd.DataFrame({'Data': True_Text_New4 , 'Prediction' : Predicted_Label_True, 'Actual': Actual_Label_True})
results_df.head(10) #Porównanie predykcji ('Prediction') z rzeczywistymi etykietami ('Actual') dla poszczególnych artykułów 


# In[43]:


X_true_New_df['Actual_Label'] = 0
Actual_Label_True = X_true_New_df['Actual_Label']

accuracy_New = accuracy_score(Actual_Label_True, new_predictions2 )
print("Accuracy:", accuracy_New)

#85% dokładności na zbiorze prawdziwych artykułów (w porównaniu do 92% na zbiorze testowym)


# In[ ]:




