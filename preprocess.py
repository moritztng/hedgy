from string import punctuation
from nltk import download, word_tokenize
from nltk.stem import PorterStemmer

download('punkt')

class StemTokenizer:
    def __init__(self):
        self.stemmer = PorterStemmer()
    def __call__(self, chapter):
        return [self.stemmer.stem(t) for t in word_tokenize(chapter) if not set(t) & set(punctuation)]
