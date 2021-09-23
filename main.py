import numpy as np
from preprocess import Vectorizer
from flask import render_template
from os.path import join, abspath, dirname
from pickle import load
from scipy.sparse import load_npz

''' def rank(transcripts, query):
    keywords = preprocess(query)
    ranking = {}
    for video_name, video_id, chapters in transcripts:
        for chapter_id, time_stamp_chapter, title, time_stamps in chapters:
            for time_stamp, words, text in time_stamps:
                missing_keywords = frozenset(keyword for keyword, keyword_stemmed in keywords if keyword_stemmed not in words)
                if len(missing_keywords) < len(keywords):
                    if (video_name, video_id, title, chapter_id, missing_keywords) in ranking:
                        ranking[(video_name, video_id, title, chapter_id, missing_keywords)].append((time_stamp, text))
                    else:
                        ranking[(video_name, video_id, title, chapter_id, missing_keywords)] = [(time_stamp, text)]
    ranking = [key + (value,) for key, value in ranking.items()]
    return sorted(ranking, key=lambda x: (len(x[-2]), -len(x[-1]))) '''

hedgy_path = dirname(abspath(__file__))
with open(join(hedgy_path, 'chapters.p'), 'rb') as chapters_f, open(join(hedgy_path, 'vectorizer.p'), 'rb') as vectorizer_f, open(join(hedgy_path, 'topics')) as topics_f:
    chapters = load(chapters_f)
    vectorizer = load(vectorizer_f)
    topics = topics_f.read().splitlines()
tfidf_matrix = load_npz(join(hedgy_path, 'tfidf.npz'))
similarity_matrix = np.load(join(hedgy_path, 'similarity.npy'))

def hedgy(request):
    ranking, sliced = [], False
    if 'query' in request.args and 'max' in request.args:
        query_vector = vectorizer.transform([request.args.get('query')])
        similarity_vector = (tfidf_matrix @ query_vector.T).toarray().squeeze()
        ranking = np.argsort(similarity_vector)[::-1][:int(request.args.get('max'))].tolist()
        sliced = True
    return render_template('hedgy.html', chapters=chapters, ranking=ranking, sliced=sliced, topics=topics, args=request.args)
