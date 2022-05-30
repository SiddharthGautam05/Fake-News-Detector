import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


df=pd.read_csv('fake_or_real_news.csv')


labels=df.label

x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)


tnews="Fear of a possible Islamic State bloodbath sent tens of thousands of Iraqis fleeing Ramadi on Monday after government forces abandoned the city -- just 80 miles from Baghdad -- in what one U.S. military official conceded was a fight 'pretty much over.'"

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)


import streamlit as st

st.title("Fake News Detector!!")
def detect():
    user=st.text_area("Enter Suspicious News Headline:")
    if len(user)<1:
        st.write(" ")
    else:
        tnews=user
        x_test[1267]=tnews
        tfidf_test=tfidf_vectorizer.transform(x_test)
        ans=pac.predict(tfidf_test)
        st.title(ans[-1])
detect()

