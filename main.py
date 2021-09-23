import numpy as np
from preprocess import Vectorizer
from flask import render_template
from os.path import join, abspath, dirname
from pickle import load
from scipy.sparse import load_npz

hedgy_path = dirname(abspath(__file__))
with open(join(hedgy_path, 'chapters.p'), 'rb') as chapters_f, open(join(hedgy_path, 'vectorizer.p'), 'rb') as vectorizer_f, open(join(hedgy_path, 'topics')) as topics_f:
    chapters = load(chapters_f)
    vectorizer = load(vectorizer_f)
    topics = topics_f.read().splitlines()
tfidf_matrix = load_npz(join(hedgy_path, 'tfidf.npz'))
similarity_matrix = np.load(join(hedgy_path, 'similarity.npy'))

def hedgy(request):
    ranking = []
    if 'query' in request.args and 'max' in request.args:
        query_vector = vectorizer.transform([request.args.get('query')])
        similarity_vector = (tfidf_matrix @ query_vector.T).toarray().squeeze()
        ranking = np.argsort(similarity_vector)[::-1][:int(request.args.get('max'))].tolist()
    return render_template('hedgy.html', chapters=chapters, ranking=ranking, topics=topics, args=request.args)
