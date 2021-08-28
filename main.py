from flask import render_template
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
    text = {(word, stemmer.stem(word)) for word in text}
    return text

def rank(transcripts, query):
    keywords = preprocess(query)
    ranking = {}
    for video_name, video_id, time_stamps in transcripts:
        for time_stamp, words, text in time_stamps:
            missing_keywords = frozenset(keyword for keyword, keyword_stemmed in keywords if keyword_stemmed not in words)
            if len(missing_keywords) < len(keywords):
                if (video_name, video_id, missing_keywords) in ranking:
                    ranking[(video_name, video_id, missing_keywords)].append((time_stamp, text))
                else:
                    ranking[(video_name, video_id, missing_keywords)] = [(time_stamp, text)]
    ranking = [key + (value,) for key, value in ranking.items()]
    return sorted(ranking, key=lambda x: (len(x[2]), -len(x[3])))

with open(join(dirname(abspath(__file__)), 'transcripts.p'), 'rb') as f:
    transcripts = load(f)

def hedgy(request):
    ranking = rank(transcripts, request.args.get('query')) if 'query' in request.args else []
    return render_template('hedgy.html', ranking=ranking, query=request.args.get('query'))
