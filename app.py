import streamlit as st
import pickle
import pandas as pd
import re
import numpy as np
import sklearn
import nltk

key_norm = pd.read_csv('key_norm.csv')

df_tempat = pickle.load(open('df_tempat.pkl','rb'))
df_tempat = pd.DataFrame(df_tempat)

df_review = pickle.load(open('df_review.pkl','rb'))
df_review = pd.DataFrame(df_review)

with open('bebas.pkl', 'rb') as f:
  P=pickle.load(f)
  Q=pickle.load(f)
  userid_vectorizer=pickle.load(f)

with open('stopword.pkl', 'rb') as f:
  stopwords_ind = pickle.load(f)
def casefolding(mess):
  mess = mess.lower()                               # Mengubah teks menjadi lower case
  mess = re.sub(r'https?://\S+|www\.\S+', '', mess) # Menghapus URL
  mess = re.sub(r'[-+]?[0-9]+', '', mess)           # Menghapus angka
  mess = re.sub(r'[^\w\s]','', mess)                # Menghapus karakter tanda baca
  mess = mess.strip()
  return mess

def remove_stop_words(mess):
  clean_words = []
  mess = mess.split()
  for word in mess:
      if word not in stopwords_ind:
          clean_words.append(word)
  return " ".join(clean_words)

def text_normalize(mess):
  mess = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if (key_norm['singkat'] == word).any() else word for word in mess.split()])
  mess = str.lower(mess)
  return mess

def text_preprocessing_process(mess):
  mess = casefolding(mess)
  mess = text_normalize(mess)
  mess = remove_stop_words(mess)
  return mess

def recommend(words):
  test_df= pd.DataFrame([words], columns=['review_text'])
  test_df['review_text'] = test_df['review_text'].apply(text_preprocessing_process)
  test_vectors = userid_vectorizer.transform(test_df['review_text'])
  test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index, columns=userid_vectorizer.get_feature_names_out())

  predictItemRating=pd.DataFrame(np.dot(test_v_df.loc[0],Q.T),index=Q.index,columns=['Rating'])
  topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:5]
  recommended_place = []
  recommended_rating = []
  recommended_address = []
  for i in topRecommendations.index:
    recommended_place.append(df_tempat[df_tempat['resto_id']==i]['name'].iloc[0])
    recommended_rating.append(df_tempat[df_tempat['resto_id'] == i]['overall_rating'].iloc[0])
    recommended_address.append(df_tempat[df_tempat['resto_id'] == i]['location'].iloc[0])
  return recommended_place,recommended_rating,recommended_address

st.title('Sistem Rekomendasi Restaurant di Bandung')
inputan = st.text_area('Kuliner seperti apa yang Anda inginkan?')

if st.button('Rekomendasi'):
    recommended_place,recommended_rating,recommended_address = recommend(inputan)
    st.header(recommended_place[0])
    st.subheader(recommended_rating[0])
    st.caption(recommended_address[0])
    st.header(recommended_place[1])
    st.subheader(recommended_rating[1])
    st.caption(recommended_address[1])
    st.header(recommended_place[2])
    st.subheader(recommended_rating[2])
    st.caption(recommended_address[2])
    st.header(recommended_place[3])
    st.subheader(recommended_rating[3])
    st.caption(recommended_address[3])
    st.header(recommended_place[4])
    st.subheader(recommended_rating[4])
    st.caption(recommended_address[4])