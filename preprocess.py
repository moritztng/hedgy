from string import punctuation
from nltk import download, word_tokenize
from nltk.stem import PorterStemmer
from numpy import float32
from sklearn.feature_extraction.text import TfidfVectorizer

download('punkt')

class StemTokenizer:
    def __init__(self):
        self.stemmer = PorterStemmer()
    def __call__(self, chapter):
        return [self.stemmer.stem(t) for t in word_tokenize(chapter) if not set(t) & set(punctuation)]

class Vectorizer(TfidfVectorizer):
    def __init__(self):
        super().__init__(tokenizer=StemTokenizer(), dtype=float32)
