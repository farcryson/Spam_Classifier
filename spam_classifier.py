import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def transform_text(text):
  text = re.sub('[^a-zA-Z]', ' ', text)
  text = text.lower()
  text = text.split()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  ps = PorterStemmer()
  text = [ps.stem(word) for word in text if not word in set(all_stopwords)]
  text = ' '.join(text)
  return text

classifier = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))

st.title("Spam Classifier")

message = st.text_area("Enter your message")

if st.button('Predict'):
  # 1. Preprocess
  transformed_message = transform_text(message)
  # 2. Vectorize
  vectorized_message = cv.transform([transformed_message])
  # 3. Predict
  result = classifier.predict(vectorized_message)
  # 4. Display
  if result == 1:
    st.header("Spam")
  else:
    st.header("Ham")