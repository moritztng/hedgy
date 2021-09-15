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
                    if (video_name, video_id, title, missing_keywords) in ranking:
                        ranking[(video_name, video_id, title, missing_keywords)].append((time_stamp, text))
                    else:
                        ranking[(video_name, video_id, title, missing_keywords)] = [(time_stamp, text)]
    ranking = [key + (value,) for key, value in ranking.items()]
    return sorted(ranking, key=lambda x: (len(x[-2]), -len(x[-1])))

hedgy_path = dirname(abspath(__file__))
with open(join(hedgy_path, 'transcripts.p'), 'rb') as transcripts_f, open(join(hedgy_path, 'topics')) as topics_f:
    transcripts = load(transcripts_f)
    topics = topics_f.read().splitlines()

def hedgy(request):
    ranking, sliced = [], False
    if 'query' in request.args:
        ranking = rank(transcripts, request.args.get('query'))
        if 'max' in request.args:
            sliced = len(ranking) > int(request.args.get('max'))
            ranking = ranking[:int(request.args.get('max'))]
    return render_template('hedgy.html', topics=topics, ranking=ranking, sliced=sliced, args=request.args)
