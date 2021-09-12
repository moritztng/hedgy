from flask import render_template
from preprocess import preprocess
from os.path import join, abspath, dirname
from pickle import load

def rank(transcripts, query):
    keywords = preprocess(query)
    ranking = {}
    for video_name, video_id, chapters in transcripts:
        for time_stamp_chapter, title, time_stamps in chapters:
            for time_stamp, words, text in time_stamps:
                missing_keywords = frozenset(keyword for keyword, keyword_stemmed in keywords if keyword_stemmed not in words)
                if len(missing_keywords) < len(keywords):
                    if (video_name, video_id, missing_keywords) in ranking:
                        ranking[(video_name, video_id, missing_keywords)].append((title, time_stamp, text))
                    else:
                        ranking[(video_name, video_id, missing_keywords)] = [(title, time_stamp, text)]
                    break
    ranking = [key + (value,) for key, value in ranking.items()]
    return sorted(ranking, key=lambda x: (len(x[2]), -len(x[3])))

with open(join(dirname(abspath(__file__)), 'transcripts.p'), 'rb') as f:
    transcripts = load(f)

def hedgy(request):
    ranking, sliced = [], False
    if 'query' in request.args:
        ranking = rank(transcripts, request.args.get('query'))
        if 'max' in request.args:
            n_time_stamps = 0
            for i, item in enumerate(ranking):
                n_time_stamps += len(item[-1])
                if n_time_stamps > int(request.args.get('max')) and i:
                    ranking, sliced = ranking[:i], True
                    break
    return render_template('hedgy.html', ranking=ranking, sliced=sliced, args=request.args)
