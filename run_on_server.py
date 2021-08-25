#Generic
import os
from os.path import join
import re
import numpy as np
import pandas as pd
import pickle
import itertools
import datetime
from datetime import timedelta

#Gensim (LDA-Modelling)
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import LdaMulticore
from gensim.test.utils import datapath

import pyLDAvis.gensim_models


#functions

def get_lda(corpus, id2word, num_topics, n_jobs=32, passes=200, chunksize=100):

    lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus, 
                                        id2word=id2word,
                                        num_topics=num_topics,
                                        random_state=100,
                                        chunksize=chunksize,
                                        workers=n_jobs, 
                                        passes=passes,
                                        per_word_topics=True)
    lda_model.save(datapath('/home/cloud/arabic_nlp/results/lda_' + str(num_topics) + '_topics'  ))
    return lda_model

def visu(lda_model, corpus, id2word, name):
    print('process visu')
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    print('vis calc')
    pyLDAvis.save_html(vis, r'/home/cloud/arabic_nlp/results/' + name + '.html')
    return


# code to process

# load preprocessed data
with open('/home/cloud/arabic_nlp/data/save_dict_full.pkl', 'rb') as f:
        data_dict = pickle.load(f)

# with open('/home/cloud/arabic_nlp/data/save_dict_samp.pkl', 'rb') as f:
#         data_dict = pickle.load(f)

with open('/home/cloud/arabic_nlp/data/comparison_list_alittihad.pkl', 'rb') as f:
        comp_list = pickle.load(f)

corpus = data_dict['corpus_full']
id2word = data_dict['id2word_full']

#print(corpus)
#print(id2word)

for n in range(8,23,3):
    print('started' + str(n))
    lda = get_lda(corpus, id2word, num_topics=n)
    # name = str(n) + '_topics'
    # visu(lda, corpus, id2word, name)

print('done!')
