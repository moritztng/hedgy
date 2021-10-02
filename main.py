import numpy as np
from preprocess import Vectorizer
from flask import render_template, make_response
from google.oauth2.id_token import verify_oauth2_token
from google.auth.transport import requests
from os.path import join, abspath, dirname
from random import randint
from pickle import load
from scipy.sparse import load_npz

hedgy_path = dirname(abspath(__file__))
with open(join(hedgy_path, 'chapters.p'), 'rb') as chapters_f, open(join(hedgy_path, 'vectorizer.p'), 'rb') as vectorizer_f:
    chapters = load(chapters_f)
    vectorizer = load(vectorizer_f)
tfidf_matrix = load_npz(join(hedgy_path, 'tfidf.npz'))
similarity_matrix = np.load(join(hedgy_path, 'similarity.npy'))

def hedgy(request): 
    ranking, sliced, max_request, seed, token = [], False, 50, None, None
    credential = request.form['credential'] if request.method == 'POST' else request.cookies.get('credential')
    if credential:
        try:
            token = verify_oauth2_token(credential, requests.Request(), '1080182836213-psdjtgo2u10a1fb6e4sbdfpdlmco5i63.apps.googleusercontent.com')
        except:
            pass
    if 'max' in request.args:
        max_request = int(request.args.get('max'))
        if 'query' in request.args or 'similar' in request.args:
            if 'query' in request.args:
                query_vector = vectorizer.transform([request.args.get('query')])
                similarity_vector = (tfidf_matrix @ query_vector.T).toarray().squeeze()
            else:
                similarity_vector = similarity_matrix[int(request.args.get('similar'))]
            if np.any(similarity_vector):
                max_chapters = np.count_nonzero(similarity_vector)
                if max_request < max_chapters:
                    max_chapters = max_request
                    sliced = True
                ranking = np.argsort(similarity_vector)[::-1][:max_chapters].tolist()
        elif 'seed' in request.args:
            seed = int(request.args.get('seed'))
    else:
        seed = randint(1, 1000000)
    if seed:
        np.random.seed(seed)
        ranking = np.random.permutation(len(chapters))[:max_request].tolist()
        sliced = True
    response = make_response(render_template('hedgy.html', chapters=chapters, ranking=ranking, sliced=sliced, max_request=max_request, seed=seed, token=token, args=request.args))
    if request.method == 'POST':
        response.set_cookie('credential', request.form['credential'], secure=True, httponly=True)
    return response
