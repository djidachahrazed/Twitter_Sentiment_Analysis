import streamlit as st
import numpy as np
import pandas as pd
import re
import tweepy
import pandas as pd
from pandas import DataFrame
from tweepy import OAuthHandler
import preprocessor as p
from PIL import Image
from bokeh.models.widgets import Div
import api_extract as pi
import fonctions as fct
import re
import tweepy
from tweepy import OAuthHandler
import os

from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
#import preprocessor as p
import requests
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# keys and tokens from the Twitter Dev Console 
consumer_key = '9sbiaw8LE9JD8z3xLm1q18gN6'
consumer_secret = 'NA7mz0NWIvzhZgwxtVqBIxorLMRLJy9VCbSqZqOG79mKddj0f2'
access_token = '1291368045592686593-k5J6P77O6GTm9GDdw1gfX6xLiXSgET'
access_token_secret = 'gMtPbmyBwgjRUrH48phJvEtbVcFCGMJFvaq1vtT4njeEQ'

# twitter authentification
try:
  # create OAuthHandler object
  auth = OAuthHandler(consumer_key, consumer_secret) 
  # set access token and secret 
  auth.set_access_token(access_token, access_token_secret) 
  # create tweepy API object to fetch tweets 
  api = tweepy.API(auth)
except:
  print("Error: Authentication Failed")

st.title('Arabic Sentiment Analysis In Twitter')

menu = ["Clean data","Sentiment", "Prediction","Classification", "Visualisation"]
choice = st.sidebar.selectbox("Menu", menu)
#home page
if choice == "Clean data":
  st.markdown(f'<div style="color: MediumVioletRed;font-weight: bold; font-size: x-large">A place where the feeback matters </div>',unsafe_allow_html=True)
  st.write("Welcome to SentArab ! A platform for teachers and students that want to improve their elearning experinece!  ")
    #/******************test***********************
  img= Image.open("C:\\Users\\CHAHRAZED\\OneDrive\\Desktop\\projetYass\\code\\senti.jpg")
  
  st.write("With SVM and Knn Classification")
  st.image(img,caption='', use_column_width=True)

  name= st.text_input("SEARCH")
  if st.button("submit"):
    data=pi.get_tweets(name)
    data=pi.get_tweets_listes(name)
    data=pi.data_to_df(name)
    st.write(data)

  if st.button("save to csv"):
    save=pi.data_to_csv(name)


elif choice == "Sentiment":
  st.markdown(f'<div style="color: MediumVioletRed;font-weight: bold; font-size: x-large"> Arabic Sentiment Using Arabic Tweets</div>',unsafe_allow_html=True)
  st.write("Welcome to Sentiment")
  def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)
  filename = file_selector()
  st.write('You selected `%s`' % filename)
  df=pd.read_csv(filename)

  if st.checkbox("show dataset"):
    nb=st.number_input("Numbers of Rows to Views",5,20)
    st.dataframe(df.head(nb))
  if st.checkbox("clean data"):
    clean=fct.cleaning(df)
    nb=st.number_input("Numbers of Rows to Views",5,20)
    st.dataframe(clean.head(nb)) 
  if st.checkbox("get users"):
    users=fct.get_user_df(df)
    nb=st.number_input("Numbers of Rows to Views",5,20)
    st.dataframe(users.head(nb))
  if st.checkbox('show sentiment'):
  	sent=fct.predict_df(df)
  	nb=st.number_input("Numbers of Rows to Views",5,20)
  	st.dataframe(sent.head(nb))
  if st.checkbox('show final data'):
    nb=st.number_input("Numbers of Rows to Views",5,20)
    df=fct.predict_df(df)
    df=fct.cleaning(df)
    st.dataframe(df.head(nb))
  if st.checkbox('save result'):
  	
  	save=df.to_csv("result.csv")


   
        
elif choice == "Prediction": 
  st.markdown(f'<div style="color: MediumVioletRed;font-weight: bold; font-size: x-large">SVM / KNN </div>',unsafe_allow_html=True)
  st.write("Prediction with API")
  
  txt= st.text_input("Your Review")
  if st.button('predict'):
    pre=fct.predict(txt)
    if pre=='positive':
      st.write(f'<div style="color:green">positive</div>',unsafe_allow_html=True)
    elif pre == 'negative':
      st.write(f'<div style="color:red">negative</div>',unsafe_allow_html=True)
    elif pre == 'neutral':
      st.write(f'<div style="color:orange">neutral</div>',unsafe_allow_html=True)


elif choice == "Classification": 
  st.markdown(f'<div style="color: MediumVioletRed;font-weight: bold; font-size: x-large">SVM / KNN </div>',unsafe_allow_html=True)
  st.write("Welcome to Classification")
  def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)
  filename = file_selector()
  st.write('You selected `%s`' % filename)
  df=pd.read_csv(filename)
  cv = CountVectorizer()
  X_data = cv.fit_transform(df['comments'].values.astype('U'))

  tfidf = TfidfTransformer()
  X_data_tfidf = tfidf.fit_transform(X_data)
  X_train, X_test, y_train, y_test = train_test_split(X_data_tfidf,df['sentiment'], test_size=0.33, random_state=42)
  st.write(X_train.shape, X_test.shape)

  if st.checkbox("show dataset"):
    nb=st.number_input("Numbers of Rows to Views",5,20)
    st.dataframe(df.head(nb))
  if st.checkbox("SVM"):
    st.write("SVM Classification")
    cv = CountVectorizer()
    X_data = cv.fit_transform(df['comments'].values.astype('U'))

    tfidf = TfidfTransformer()
    X_data_tfidf = tfidf.fit_transform(X_data)
    X_train, X_test, y_train, y_test = train_test_split(X_data_tfidf,df['sentiment'], test_size=0.33, random_state=42)
    st.write(X_train.shape, X_test.shape)
    #svm
    svc = SVC()
    svc.fit(X_train, y_train)

    preds = svc.predict(X_test)
    acc = np.mean(preds == y_test)
    st.write('SVC model accuracy: {}'.format(acc*100))

    cr = classification_report(y_test, svc.predict(X_test))
    st.write(cr)
  if st.checkbox("KNN"):
  	st.write("KNN Classification")
  	knn_9 = KNeighborsClassifier(n_neighbors=3)
  	knn_9.fit(X_train, y_train)
  	# predict on the test-set
  	y_pred_9 = knn_9.predict(X_test)
  	st.write('Model accuracy score with k=3 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_9)))
  	st.write(classification_report(y_test, y_pred_9))


        
elif choice == "Visualisation":
  st.markdown(f'<div style="color: MediumVioletRed;font-weight: bold; font-size: x-large">Data Visualisation</div>',unsafe_allow_html=True)
  st.write("Welcome to Visualisation")
  def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)
  filename = file_selector()
  st.write('You selected `%s`' % filename)
  df=pd.read_csv(filename)

  #plot
  if st.button("plot"):
  	df.sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%')
  	st.pyplot()

        
