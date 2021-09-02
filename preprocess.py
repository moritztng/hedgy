from string import punctuation
from nltk import word_tokenize, download
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

download('stopwords')
download('punkt')

def preprocess(text):
    text = text.lower()
    text = set(word_tokenize(text))
    text = {word for word in text if not set(word) & set(punctuation)}
    text = text - set(stopwords.words('english'))
    stemmer = PorterStemmer()
    text = {(word, stemmer.stem(word)) for word in text}
    return text
