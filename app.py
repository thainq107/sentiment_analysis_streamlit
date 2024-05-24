import os
import gdown
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_model(gdrive_id='1-0VD3Bk4y6mUyIttBt00naeRYtrjD8L5'):

  model_path = 'distilbert-imdb'
  if not os.path.exists(model_path):
    # download folder
    gdown.download_folder(id=gdrive_id)
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForSequenceClassification.from_pretrained(model_path)
  return tokenizer, model

tokenizer, model = load_model()
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def main():
  st.title('Sentiment Analysis')
  st.title('Model: Distil BERT. Dataset: IMDB-Review')
  text_input = st.text_input("Sentence: ", "I like this movie!")
  result = classifier(text_input)
  st.success(result[0]['label']) 

if __name__ == '__main__':
     main() 
  
