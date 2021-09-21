import numpy as np
from flask import render_template
from preprocess import preprocess
from os.path import join, abspath, dirname
from pickle import load

def rank(transcripts, query):
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
    return sorted(ranking, key=lambda x: (len(x[-2]), -len(x[-1])))

hedgy_path = dirname(abspath(__file__))
with open(join(hedgy_path, 'transcripts.p'), 'rb') as transcripts_f, open(join(hedgy_path, 'topics')) as topics_f:
    transcripts = load(transcripts_f)
    topics = topics_f.read().splitlines()
similarity_matrix = np.load(join(hedgy_path, 'similarity.npy'))

def hedgy(request):
    ranking, sliced = [], False
    if 'query' in request.args:
        ranking = rank(transcripts, request.args.get('query'))
        if 'max' in request.args:
          sliced = len(ranking) > int(request.args.get('max'))
          ranking = ranking[:int(request.args.get('max'))]
    elif 'similar' in request.args:
        sliced = True
        ranking = np.argsort(similarity_matrix[int(request.args.get('similar'))])[::-1][:int(request.args.get('max'))].tolist()
        for video_name, video_id, chapters in transcripts:
            for chapter_id, time_stamp_chapter, title, time_stamps in chapters:
                if chapter_id in ranking:
                    ranking[ranking.index(chapter_id)] = (video_name, video_id, title, chapter_id, False, [(time_stamps[0][0], time_stamps[0][2])])
    return render_template('hedgy.html', topics=topics, ranking=ranking, sliced=sliced, args=request.args)
