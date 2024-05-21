import gdown
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

@st.cache
def load_pipeline(gdrive_id='1-0VD3Bk4y6mUyIttBt00naeRYtrjD8L5'):
  # download folder
  gdown.download_folder(id=gdrive_id)

  model_path = 'distilbert-imdb
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForSequenceClassification.from_pretrained(model_path)
  return tokenizer, model

tokenizer, model = load_pipeline()
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def main():
  st.title('Sentiment Analysis')
  st.title('Model: Distil BERT. Dataset: IMDB-Review')
  text_input = st.text_input("Sentence: ", "I like this movie!")
  result = classifier(text_input)
  st.success(result[0]['label']) 

if __name__ == '__main__':
     main() 
  
