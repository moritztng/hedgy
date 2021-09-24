import numpy as np
from preprocess import Vectorizer
from flask import render_template
from os.path import join, abspath, dirname
from pickle import load
from scipy.sparse import load_npz

hedgy_path = dirname(abspath(__file__))
with open(join(hedgy_path, 'chapters.p'), 'rb') as chapters_f, open(join(hedgy_path, 'vectorizer.p'), 'rb') as vectorizer_f:
    chapters = load(chapters_f)
    vectorizer = load(vectorizer_f)
tfidf_matrix = load_npz(join(hedgy_path, 'tfidf.npz'))
similarity_matrix = np.load(join(hedgy_path, 'similarity.npy'))

def hedgy(request):
    ranking, sliced = [], False
    if 'max' in request.args and ('query' in request.args or 'similar' in request.args):
        if 'query' in request.args:
            query_vector = vectorizer.transform([request.args.get('query')])
            similarity_vector = (tfidf_matrix @ query_vector.T).toarray().squeeze()
        else:
            similarity_vector = similarity_matrix[int(request.args.get('similar'))]
        if np.any(similarity_vector):
            max_chapters, max_request = np.count_nonzero(similarity_vector), int(request.args.get('max'))
            if max_request < max_chapters:
                max_chapters = max_request
                sliced = True
            ranking = np.argsort(similarity_vector)[::-1][:max_chapters].tolist()
    else:
        ranking = np.random.permutation(len(chapters))[:50].tolist()
    return render_template('hedgy.html', chapters=chapters, ranking=ranking, sliced=sliced, args=request.args)
