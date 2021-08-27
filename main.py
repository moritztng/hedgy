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
    text = {stemmer.stem(word) for word in text}
    return text

def rank(transcripts, query):
    keywords = preprocess(query)
    ranking = []
    for transcript in transcripts:
        for time_stamp, words, text in transcript[2]:
            rank = len(words & keywords)
            if rank:
                ranking.append(transcript[:2] + [time_stamp, text, rank])
    ranking.sort(key=lambda x: x[-1], reverse=True)
    ranking_grouped = []
    for video_name, video_id, time_stamp, text, _ in ranking:
        if ranking_grouped and ranking_grouped[-1][1] == video_id:
            ranking_grouped[-1][-1].append([time_stamp, text]) 
        else:
            ranking_grouped.append([video_name, video_id, [[time_stamp, text]]])
    return ranking_grouped

with open(join(dirname(abspath(__file__)), 'transcripts.p'), 'rb') as f:
    transcripts = load(f)

def hedgy(request):
    ranking = rank(transcripts, request.args.get('query')) if 'query' in request.args else []
    return render_template('hedgy.html', ranking=ranking, query=request.args.get('query'))
