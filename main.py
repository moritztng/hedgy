from string import punctuation
from nltk import word_tokenize, download
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from os.path import join, abspath, dirname
from pickle import load

download('stopwords')
download('punkt')

def preprocess(text):
    text = text.lower()
    text = set(word_tokenize(text))
    text = {word for word in text if not set(word) & set(punctuation)}
    text = text - set(stopwords.words('english'))
    stemmer = PorterStemmer()
    text = {stemmer.stem(word) for word in text}
    return text

def rank(transcripts, query):
    keywords = preprocess(query)
    ranking = []
    for transcript in transcripts:
      for time_stamp, words in transcript[2]:
        rank = len(words & keywords)
        if rank:
          ranking.append(transcript[:2] + [time_stamp, rank])
    return sorted(ranking, key=lambda x: x[-1], reverse=True)

with open(join(dirname(abspath(__file__)), 'transcripts.p'), 'rb') as f:
    transcripts = load(f)

def hedgy(request):
    response = """<h1>Search Lex!</h1>
                  <form action='' method='get'>
                    <input type='search' name='query' required>
                    <input type='submit' value='Search'>
                  </form>"""
    if 'query' in request.args:
        ranking = rank(transcripts, request.args.get('query'))
        for time_stamp in ranking:
            response += f"<a href='https://youtu.be/{time_stamp[1]}?t={time_stamp[2]}'>{time_stamp[0]}</a><br>\n"
    return response
