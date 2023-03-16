from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import plotly.express as px

import matplotlib.pyplot as plt

st.header('Sentiment Analysis')
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        x= round(blob.sentiment.polarity,2)
        if x >= 0.5:
           st.write('Sentiment: Positive')
        elif x <= -0.5:
            st.write('Sentiment: Negative')
        else:
             st.write('Sentiment: Neutral')


    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True ,stopwords=True ,lowercase=True ,numbers=True , punct=True))

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

#
    def analyze(x):
        if x >0:
            return 'Positive'
        elif x <0:
            return 'Negative'
        else:
            return 'Neutral'

#
    if upl:
        df = pd.read_excel(upl)
        del df['Unnamed: 0']
        df['score'] = df['tweets'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        pos=(df['analysis']=='Positive').sum()
        neg=(df['analysis']=='Negative').sum()
        neu=(df['analysis']=='Neutral').sum()
        labels='Positive','Negative','Neutral'
        values=[pos,neg,neu]
        fig1, ax1 = plt.subplots()
        explode = (0, 0.1, 0.1) 
        ax1.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
        ax1.axis('equal') 
        st.pyplot(fig1)
        #l=len(df)
        #st.write(df.head(l-1))
        
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(df)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )
