# -*- coding: utf-8 -*-
"""LDA2Vec (HW 2 Part 3).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jy_poDlIf7BBQRTQPGw55ZTp0BhhA7fq

# Hw2 Part 3
# LDA2Vec
"""

from google.colab import drive
drive.mount('/content/gdrive')

import os
path = r'/content/gdrive/My Drive/Colab_Datasets'
os.chdir(path)

# Commented out IPython magic to ensure Python compatibility.
# Run in terminal or command prompt to download spacy dict
# python3 -m spacy download en
# copyright Felipe Castrollio
!pip install nltk
nltk.download('punkt')
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
import math

# topics library
#!pip install topics
#from topics import prepare_topics
#from topics import print_top_words_per_topic

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint


from nltk.tokenize import word_tokenize
# Plotting tools
!pip install pyLDAvis
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
# %matplotlib inline

# Gensim
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models import ldaseqmodel
from gensim.test.utils import datapath


# Keras tools
import keras
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Dot
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras import layers
from keras.engine.input_layer import Input
from keras import backend as K
import tensorflow as tf

from datetime import datetime, date

#!pip install playsound
#from playsound import playsound

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

"""### Model Parameters """

# variables
window_size = 10
epochs      = 50000  # 200000
n_topics    = 12     # for lda model
vector_dim  = 100
batch_size  = 1000

# validation 
valid_size     = 16     # Random set of words to evaluate similarity on.
valid_window   = 50     # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

"""### Data PreProcessing

##### Regular Sampling
"""

df = pd.read_excel('Index_fund_all.xlsx')

"""##### Stratified Sampling """

df = pd.read_excel('Index_fund_all.xlsx')
df = df.dropna(axis=0)
n = 10
df = df.groupby('filing_year').apply(lambda x: x.sample(n=n))    #[389, 504, 579, 619, 628, 636, 705, 810, 891]
df = df.reset_index(drop=True)
df = pd.DataFrame(df)

#Refer to https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/ 
#for explanation on data processing steps below.
# Convert to list
data = df.principal_strategies.values.tolist()

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', str(sent)) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', str(sent)) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", str(sent)) for sent in data]

#data[:1]

#-----------------------------------------------------------
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))
#print(data_words[:1])
#-----------------------------------------------------------

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out


# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# Run in terminal: python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#print(len(data_lemmatized))
#print(data_lemmatized[0])
#-----------------------------------------------------------

#tokenize every doc
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data_lemmatized)
sequences = tokenizer.texts_to_sequences(data_lemmatized)

n_documents = len(sequences)  # <<-- 

dictionary = tokenizer.word_index
dictionary["null"] = 0



reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))   #<<--
vocab_size         = len(dictionary) #<<--
#print(n_documents) = 6171
#vocab_size = 3893
#-----------------------------------------------------------



# create dataset: word pairs and doc ids with positive and negative samples
window_size = 2
targets = []
contexts = []
labels = []
couples = []
doc_ids = []

for i in range(0,n_documents):
    if i % 1000 == 0 and i > 0:
        print (i)
    seq = sequences[i]
    sampling_table = sequence.make_sampling_table(vocab_size)
    couple, label = skipgrams(seq, 
                              vocab_size, 
                              window_size=window_size, 
                              sampling_table=sampling_table)
    if not couple:
        next
    try:
        target, context = zip(*couple)
        targets = targets + list(target)
        contexts = contexts + list(context)
        doc_ids = doc_ids + [i]*len(context)
        labels = labels + label
        couples = couples + couple
    except:
        print ("Error on " + str(seq))
    
data_target  = np.array(targets, dtype='int32')
data_context = np.array(contexts, dtype='int32')
doc_ids      = np.array(doc_ids, dtype='int32')
labels       = np.array(labels, dtype='int32')


#-----------------------------------------------------------
# split into train and test
from random import sample

training_split = 0.8
l              = len(data_target)        #length of data 
f              = int(l * training_split) #number of elements you need
indices        = sample(range(l),f)

train_data_target  = data_target[indices]
test_data_target   = np.delete(data_target,indices)
train_data_context = data_context[indices]
test_data_context  = np.delete(data_context,indices)
train_doc_ids      = doc_ids[indices]
test_doc_ids       = np.delete(doc_ids,indices)
train_labels       = labels[indices]
test_labels        = np.delete(labels,indices)

#print(couples[:10], labels[:10], doc_ids[:10])

print("size of training data " + str(len(train_data_target)))
print("size of testing data " + str(len(test_data_target)))
print("size of labels " + str(len(labels)))

doc_lengths = []
for n in sequences: 
    doc_lengths.append(len(n))

print(doc_lengths)

print(len(dictionary))
print(len(corpus))
print(len(dictionary1))
print(corpus)

# print(doc_ids[indices].shape)

def lemmatization1(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags])
    return texts_out

# dl = data_words.copy()
dl2 = lemmatization1(dl2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
dictionary1 = corpora.Dictionary(dl2)
texts = dl2.copy()
corpus = [dictionary1.doc2bow(text) for text in texts]



dl2 = []
for text in texts: 
    for w in text:
        dl2.append(w)
print(dl2)

print(texts)

"""## Modeling

This is where we start creating the model. 

The model consists of two parallel flows: word embedding (like word2vec) and topic embedding (like LDA). 

Please refer to the model image here: https://github.com/cemoody/lda2vec. 

You can see on the left the word embedding happens, and on the right the topic lda embedding happens. 

At the bottom the two vectors are added together to form the final context_vector. 

The model will have three training inputs: 
    1) 

1.   input_context: pivot word
2.   input_target: word that we are trying to predict
3. input_doc: document id 


And one training output:

1.   label: 0 or 1 which defines if input_context and input_target are similar taking into account input_doc 


    
The model predictions are gien by "preds" which will output a similarity score between 0 to 1

### Building Model Functions
"""

# dir(embedding)

# x1 = embedding.get_weights()

# x1=np.array(x1[0])
# print(x1.shape)

# create input placeholder variables
input_target  = Input((1,))
input_context = Input((1,))
input_doc     = Input((1,), dtype='int32')
labels        = Input((1,))

# create word2vec layers
embedding = layers.Embedding(vocab_size, 
                             vector_dim, 
                             name='embedding')

target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)

word_context = embedding(input_context)
word_context = Reshape((vector_dim, 1))(word_context)

# create lda layers
scalar = 1 / np.sqrt(n_documents + n_topics)

all_doc_topics_embedding =(tf.Variable(tf.random_normal([n_documents, n_topics], mean=0, stddev=50*scalar), 
                                       name="doc_embeddings", 
                                       trainable=True))  # Gaussian distribution

word_embedding =(tf.Variable(tf.random_normal([vocab_size, vector_dim], mean=0, stddev=50*scalar), 
                                       name="word_embeddings", 
                                       trainable=True))

def embedding_lookup(x):
    ind = tf.cast(x, 
                  tf.int32)
    
    return tf.nn.embedding_lookup(all_doc_topics_embedding,
                                  ind,
                                  partition_strategy='mod',
                                  name="doc_proportions")

doc_topics      = keras.layers.Lambda(embedding_lookup)(input_doc)

doc_topics_norm = keras.layers.Activation(activation="softmax")(doc_topics)

transform = keras.layers.Dense(vector_dim, 
                               activation=None, 
                               use_bias=True, 
                               kernel_initializer='glorot_uniform', 
                               bias_initializer='zeros', 
                               kernel_regularizer=None, 
                               bias_regularizer=None, 
                               activity_regularizer=None, 
                               kernel_constraint=None, 
                               bias_constraint=None)

topic_context = transform(doc_topics_norm)

topic_context = Reshape((vector_dim, 1))(topic_context)

# combine context layers
context = keras.layers.Add()([word_context, topic_context])

# now perform the dot product operation to get a similarity measure between target and context
similarity = layers.dot([target, context], axes=1, normalize=True)
similarity = Reshape((1,))(similarity)

# add the sigmoid output layer
preds = Dense(1, activation='sigmoid', name='similarity')(similarity)

# defnie custom loss functions

# lda loss model
lmbda = 1.0
fraction = 1/100000
alpha = None # defaults to 1/n_topics


def dirichlet_likelihood(weights, alpha=None):
    
    num_topics = n_topics
    
    if alpha is None:
        alpha = 1 / num_topics

    log_proportions = tf.nn.log_softmax(weights)

    loss = (alpha - 1) * log_proportions

    #return -tf.reduce_sum(loss) # log-sum-exp
    return tf.reduce_sum(loss) # log-sum-exp

def loss_lda(y_pred, y_true, topics_layer):
    return lmbda*fraction*dirichlet_likelihood(topics_layer)

def loss_word2vec(y_pred, y_true):
    #return tf.math.add(tf.math.multiply(y_true, (-tf.math.log(y_pred))), 
    #                   tf.math.multiply((1 - y_true),(-tf.math.log(1 - y_pred))))
    return keras.losses.binary_crossentropy(y_true, y_pred)
    
# lda2vec loss
def loss_sum(y_pred, y_true, topics_layer):
    word2vec_loss = loss_word2vec(y_pred, y_true)
    lda_loss = loss_lda(y_pred, y_true, topics_layer)
    sum_loss = word2vec_loss + lda_loss
    return sum_loss


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

"""### Fitting Model with Tensor Flow  """

batch_size       = 150
train_loss_every = 500
test_loss_every  = 10000
starttime        = datetime.now()

# define loss functions to compute
loss = loss_sum(preds, 
                labels, 
                all_doc_topics_embedding)

loss_topics = loss_lda(preds, 
                       labels, 
                       all_doc_topics_embedding)

loss_words = loss_word2vec(preds, 
                           labels)

# define gradient descent and initialize variables
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

init_op    = tf.global_variables_initializer()

sess       = K.get_session() # get session from keras

sess.run(init_op)

# Run training loop
with sess.as_default():
    for i in range(200000):
        
        idx = np.random.randint(0, 
                                len(train_labels)-1, 
                                batch_size).tolist()
        # training happens here
        losses = sess.run([train_step, loss, loss_topics, loss_words, preds, labels], 
                          
                          feed_dict = {input_target:np.reshape(train_data_target[idx],
                                                              (-1,1)),
                                       
                                       input_context:np.reshape(train_data_context[idx],
                                                               (-1,1)),
                                       
                                       input_doc:np.reshape(train_doc_ids[idx],
                                                           (-1,1)),
                                       
                                       labels: np.reshape(train_labels[idx],
                                                         (-1,1))})
            
        # print training loss
        if i % train_loss_every == 0:
            print("Iteration {}, average sum_loss={}, average lda_loss={}, average w2v_loss={}".format(i,
                                                                                                       np.mean(losses[1]),
                                                                                                       np.mean(losses[2]),
                                                                                                       np.mean(losses[3])))
        
        # print test\loss and similar words
        if i % test_loss_every == 0:
            test_loss = sess.run([loss], 
                                 feed_dict= {input_target: np.reshape(test_data_target,
                                                                     (-1,1)),
                                             input_context: np.reshape(test_data_context,
                                                                      (-1,1)),
                                             input_doc: np.reshape(test_doc_ids,
                                                                  (-1,1)),
                                             labels: np.reshape(test_labels,
                                                                (-1,1))})
            
            print("\n\n******Iteration {}, test_loss={}****\n\n".format(i, 
                                                                        np.mean(test_loss[0])))
            
            #w_sim_cb.run_sim()
            t_sim_cb.run_sim(transform.get_weights()[0])
            
            for word in find_similar_words:
                sw_sim_cb.run_sim(word)

                
#dictionary['market']
stoptime = datetime.now()
runtime = stoptime - starttime
print('Runtime:{}'.format(runtime))
#playsound("COD.mp3")

def prepare_topics(weights, factors, word_vectors, vocab, temperature=1.0,
                   doc_lengths=doc_lengths, term_frequency=corpus, normalize=False):
    """ Collects a dictionary of word, document and topic distributions.
    Arguments
    ---------
    weights : float array
        This must be an array of unnormalized log-odds of document-to-topic
        weights. Shape should be [n_documents, n_topics]
    factors : float array
        Should be an array of topic vectors. These topic vectors live in the
        same space as word vectors and will be used to find the most similar
        words to each topic. Shape should be [n_topics, n_dim].
    word_vectors : float array
        This must be a matrix of word vectors. Should be of shape
        [n_words, n_dim]
    vocab : list of str
        These must be the strings for words corresponding to
        indices [0, n_words]
    temperature : float
        Used to calculate the log probability of a word. Higher
        temperatures make more rare words more likely.
    doc_lengths : int array
        An array indicating the number of words in the nth document.
        Must be of shape [n_documents]. Required by pyLDAvis.
    term_frequency : int array
        An array indicating the overall number of times each token appears
        in the corpus. Must be of shape [n_words]. Required by pyLDAvis.
    Returns
    -------
    data : dict
        This dictionary is readily consumed by pyLDAVis for topic
        visualization.
    """
    # Map each factor vector to a word
    topic_to_word = []
    msg = "Vocabulary size did not match size of word vectors"
    assert len(vocab) == word_vectors.shape[0], msg
    if normalize:
        word_vectors /= np.linalg.norm(word_vectors, axis=1)[:, None]
    # factors = factors / np.linalg.norm(factors, axis=1)[:, None]
    for factor_vector in factors:
        factor_to_word = prob_words(factor_vector, word_vectors,
                                    temperature=temperature)
        topic_to_word.append(np.ravel(factor_to_word))
    topic_to_word = np.array(topic_to_word)
    msg = "Not all rows in topic_to_word sum to 1"
    assert np.allclose(np.sum(topic_to_word, axis=1), 1), msg
    
    # Collect document-to-topic distributions, e.g. theta
    doc_to_topic = _softmax_2d(weights)
    msg = "Not all rows in doc_to_topic sum to 1"
    assert np.allclose(np.sum(doc_to_topic, axis=1), 1), msg
    data = {'topic_term_dists': topic_to_word,
            'doc_topic_dists': doc_to_topic,
            'doc_lengths': doc_lengths,
            'vocab': vocab,
            'term_frequency': term_frequency}
    return data



def print_top_words_per_topic(data, top_n=30, do_print=True):
    """ Given a pyLDAvis data array, print out the top words in every topic.
    Arguments
    ---------
    data : dict
        A dict object that summarizes topic data and has been made using
        `prepare_topics`.
    """
    msgs = []
    lists = []
    for j, topic_to_word in enumerate(data['topic_term_dists']):
        top = np.argsort(topic_to_word)[::-1][:top_n]
        prefix = "Top words in topic %i " % j
        top_words = [data['vocab'][i].strip().replace(' ', '_') for i in top]
        msg = ' '.join(top_words)
        if do_print:
            print (prefix + msg)
        lists.append(top_words)
    return lists

import numpy as np
import requests
import multiprocessing


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def _softmax_2d(x):
    y = x - x.max(axis=1, keepdims=True)
    np.exp(y, out=y)
    y /= y.sum(axis=1, keepdims=True)
    return y


def prob_words(context, vocab, temperature=1.0):
    """ This calculates a softmax over the vocabulary as a function
    of the dot product of context and word.
    """
    dot = np.dot(vocab, context)
    prob = _softmax(dot / temperature)
    return prob


def get_request(url):
    for _ in range(5):
        try:
            return float(requests.get(url).text)
        except:
            pass
    return None


def topic_coherence(lists, services=['ca', 'cp', 'cv', 'npmi', 'uci',
                                     'umass']):
    """ Requests the topic coherence from AKSW Palmetto
    Arguments
    ---------
    lists : list of lists
        A list of lists with one list of top words for each topic.
    >>> topic_words = [['cake', 'apple', 'banana', 'cherry', 'chocolate']]
    >>> topic_coherence(topic_words, services=['cv'])
    {(0, 'cv'): 0.5678879445677241}
    """
    url = u'http://palmetto.aksw.org/palmetto-webapp/service/{}?words={}'
    reqs = [url.format(s, '%20'.join(top[:10])) for s in services for top in lists]
    pool = multiprocessing.Pool()
    coherences = pool.map(get_request, reqs)
    pool.close()
    pool.terminate()
    pool.join()
    del pool
    args = [(j, s, top) for s in services for j, top in enumerate(lists)]
    ans = {}
    for ((j, s, t), tc) in zip(args, coherences):
        ans[(j, s)] = tc
    return ans

def filter_bad_tweets(data):
    bad = 0
    doc_topic_dists_filtered = []
    doc_lengths_filtered = []

    for x,y in zip(data['doc_topic_dists'], data['doc_lengths']):
        if np.sum(x)==0:
            bad+=1
        elif np.sum(x) != 1:
            bad+=1
        elif np.isnan(x).any():
            bad+=1
        else:
            doc_topic_dists_filtered.append(x)
            doc_lengths_filtered.append(y)

    data['doc_topic_dists'] = doc_topic_dists_filtered
    data['doc_lengths'] = doc_lengths_filtered




x1 = embedding.get_weights() #word vectors
x1 = np.array(x1[0])
z = transform.get_weights()[0] # factors 
w = all_doc_topics_embedding.eval(session=sess) # wieghts
d = dictionary1 #reverse_dictionary.values() # vocab
data = prepare_topics(w, z, x1,d)

filter_bad_tweets(data)
x2 = pyLDAvis.prepare(**data)
pyLDAvis.display(x2)

print(data)



"""### Misc.  """

print(preds)

x2 = reverse_dictionary.values()
len(x2)

print(context)
print(preds)
print(similarity)

dir(sess)

print(labels.eval(session=sess))

print()

print(doc_topics_norm)

print(target.eval(session=sess))

print(topic_context.value_index())

print(doc_topics_norm.eval(session=sess))

weights = all_doc_topics_embedding.eval(session=sess)
print(weights.shape)
#print(weights.type)
weights.max(axis=1, keepdims=True)

print(losses)

# # create evaluation models which are used to print out similar words during training.
# # This is not needed for model training, but is used to check model outputs periodically to see if model is working

# topic_context = Input(shape=(vector_dim, ))
# topic_similarity = layers.dot([topic_context, word_context], axes=1)
# topic2words_model = Model(input=[topic_context,input_context], output=topic_similarity)

# words_similarity = layers.dot([target, word_context], axes=1, normalize=True)
# nearby_words_model = Model(input=[input_target, input_context], output=words_similarity)

# find_similar_words =['144a']

# # Evaluation functions to print similar words given a topic and similar words given another word
# # This is not used for training, but for periodic evaluation of the model

# class TopicSimilarityCallback:
#     def run_sim(self, topics):
#         for i in range(n_topics):
#             top_k = 8  # number of nearest neighbors
#             sim = self._get_sim(topics[i])
#             nearest = (-sim).argsort()[0:top_k + 1]
#             log_str = 'Closest words to topic %d:' % i
#             for k in range(top_k):
#                 close_word = reverse_dictionary[nearest[k]]
#                 log_str = '%s %s,' % (log_str, close_word)
#             print(log_str)
    
#     @staticmethod
#     def _get_sim(topic):
#         sim = np.zeros((vocab_size,))
#         in_arr1 = np.reshape(topic,(1,-1))
#         in_arr2 = np.zeros((1,))
#         for i in range(vocab_size):
#             in_arr2[0,] = i
#             out = topic2words_model.predict_on_batch([in_arr1, in_arr2])
#             sim[i] = out
#         return sim
# t_sim_cb = TopicSimilarityCallback()


# class WordsSimilarityCallback:
#     def run_sim(self):
#         for i in range(valid_size):
#             valid_word = reverse_dictionary[valid_examples[i]]
#             top_k = 10  # number of nearest neighbors
#             sim = self._get_sim(valid_examples[i])
#             nearest = (-sim).argsort()[0:top_k + 1]
#             log_str = 'Nearest to %s:' % valid_word
#             for k in range(top_k):
#                 close_word = reverse_dictionary[nearest[k]]
#                 log_str = '%s %s,' % (log_str, close_word)
#             print(log_str)

#     @staticmethod
#     def _get_sim(valid_word_idx):
#         sim = np.zeros((vocab_size,))
#         in_arr1 = np.zeros((1,))
#         in_arr2 = np.zeros((1,))
#         in_arr1[0,] = valid_word_idx
#         for i in range(vocab_size):
#             in_arr2[0,] = i
#             out = nearby_words_model.predict_on_batch([in_arr1, in_arr2])
#             sim[i] = out
#         return sim
# w_sim_cb = WordsSimilarityCallback()

# class SpecificWordsSimilarityCallback:
#     def run_sim(self, word):
#         if word not in dictionary:
#             print('Nearest to %s: Word does not exist in dictionary' % word)
#             return
#         word_index = dictionary[word]
#         top_k = 10  # number of nearest neighbors
#         sim = self._get_sim(word_index)
#         nearest = (-sim).argsort()[0:top_k + 1]
#         log_str = 'Nearest to %s:' % word
#         for k in range(top_k):
#             close_word = reverse_dictionary[nearest[k]]
#             log_str = '%s %s,' % (log_str, close_word)
#         print(log_str)

#     @staticmethod
#     def _get_sim(valid_word_idx):
#         sim = np.zeros((vocab_size,))
#         in_arr1 = np.zeros((1,))
#         in_arr2 = np.zeros((1,))
#         in_arr1[0,] = valid_word_idx
#         for i in range(vocab_size):
#             in_arr2[0,] = i
#             out = nearby_words_model.predict_on_batch([in_arr1, in_arr2])
#             sim[i] = out
#         return sim
# sw_sim_cb = SpecificWordsSimilarityCallback()