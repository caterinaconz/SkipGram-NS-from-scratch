# shift alt A to comment a chunk of code on VScode

# loading dependencies
from __future__ import division
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# useful stuff
import numpy as np
import random as rd
from scipy.special import expit
from sklearn.preprocessing import normalize
import string

__authors__ = ['Caterina Conz','Yago Bardi Vale','Aprajita Arora', 'Fatmanur Sever']
__emails__  = ['caterina.conz@student-cs.fr','yago.bardi@student-cs.fr','Aprajita.Arora@student-cs.fr', 'Fatmanur.Sever@student-cs.fr']


def text2sentences(path):
    sentences = []
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
    with open(path, encoding='utf8') as f:
        for l in f:
            sentence = l.translate(str.maketrans("","", punctuation))
            sentences.append(sentence.lower().split())
    return sentences

def loadPairs(path):
    """Loads the pairs of words from the test set"""
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

def words(sentences, minCount = 4):
    #calculate frequency of each word in the whole vocab
    n_words = {}
    frequent = {}
    idxs = {}
    counter = 0
    
    for sentence in sentences:
        for word in sentence:
            if word in n_words:
                n_words[word] += 1
                if n_words[word] >= minCount:
                    frequent[word] = n_words[word]
            else:
                n_words[word] = 1
                idxs[word] = counter
                counter += 1
    return n_words, frequent, idxs

def probs_table(frequents, idxs):
    #probability of each word from vocab to be picked for negative sampling extracted from unigram distribution
    
    # mappa ogni parola con un valore di probabilità
    
    idx = []
    table_size = len(frequents)
    prob = []
    words_list = []
    total_frequences = sum([i**(3/4) for i in list(frequents.values())]) #maybe to scaling?
    for word in frequents:
        proba = frequents[word]**(3/4)/total_frequences
        prob.append(proba*table_size)
        idx += [idxs[word] for i in range(int(proba)+2)]
    return idx

def subsampling( n_words, t =0.00001, thresh = 0.9):
    # subsampling of the frequent words to remove stopwords
    dist_words = np.array(list(n_words.values()))
    dist_words = dist_words/np.sum(dist_words)
    distribution = (dist_words-t)/dist_words - np.sqrt(t/dist_words)
    idx_subsample = np.where(distribution > thresh)
    idx2words = np.array(list(n_words.keys()))
    return idx2words[idx_subsample]

def sigmoid(a,b):
    sig = 1/(1+np.exp(-a @ b.T))
    return sig.reshape((-1,1))
    



class SkipGram:
    def __init__(self, sentences, sub_sampling=False, vector_norm = False, nEmbed=100, negativeRate=5, winSize = 3, minCount = 5, lr = 1e-1, epochs = 30, display_rate = 10):

        vocabulary, frequent_words, indexes = words(sentences)
        if sub_sampling: # if subsampling=True
          stop_words = subsampling(vocabulary)  #
          sentences = [list(filter(lambda word: not word in stop_words, sentence)) for sentence in sentences] # selects only the words NOT present in stop words
          vocabulary, frequent_words, indexes = words(sentences)

        self.negativeRate = negativeRate
        self.lr = lr
        self.winSize = winSize
        self.epochs = epochs
        self.display_rate = display_rate
        self.vector_norm = vector_norm
        self.nEmbed = nEmbed
        
        self.loss = []
        self.positive_pairs = []
        self.trainWords = 0  # was 0, changed that to 1 since was giving zero division error
        self.accLoss = 0


        self.w2id = indexes # word to ID mapping
        self.id2w = {self.w2id[w]:w for w in indexes.keys()}  # takes each id and maps it into the words, the opposite of before
        self.trainset = sentences # set of sentences
        self.vocab = vocabulary # list of valid words
        self.index_table = probs_table( frequent_words, indexes)

        self.W = np.random.uniform(-0.5, 0.5, size=(len(self.vocab),
                                                    nEmbed))  # intializing weights matrix from a uniform (W= input layer, C =weight of the output layer)
        self.C = np.random.uniform(-0.5, 0.5, size=(len(self.vocab), nEmbed))

        
    def sample(self, omit):
        #negative words sampling
        negative_words = []
        N = len(self.index_table)
        while len(negative_words) < self.negativeRate:
            neg_word_ind = self.index_table[rd.randint(0,N-1)]
            if neg_word_ind not in omit:
                negative_words.append(neg_word_ind)
        return np.array(negative_words)

    def train(self):
        best_loss = - np.inf
        count_loss = 0
        for epoch in range(self.epochs):
            np.random.shuffle(self.trainset)
            for sentence in self.trainset:
                for wpos, word in enumerate(sentence):
                    wIdx = self.w2id[word]
                    winsize = np.random.randint(self.winSize) + 1
                    start = max(0, wpos - winsize)
                    end = min(wpos + winsize + 1, len(sentence))
                    for context_word in sentence[start:end]:
                        ctxtId = self.w2id[context_word]
                        if ctxtId == wIdx: 
                            continue
                        else:
                            negativeIds = self.sample({wIdx,ctxtId})
                            self.trainWord(wIdx, ctxtId, negativeIds)
                            self.accLoss += self.compute_objective_function(wIdx, ctxtId, negativeIds)
                            self.trainWords += 1
            loss_epoch = self.accLoss / self.trainWords
            self.loss.append(loss_epoch)
            if loss_epoch > best_loss:
                best_loss = loss_epoch
                count_loss = 0
            else:
                count_loss +=1
            if count_loss > 5:
                self.lr /=10
                count_loss = 0
            if self.lr < 1e-6:
                break
            self.trainWords = 0
            self.accLoss = 0
        if self.vector_norm:
            self.W /= np.linalg.norm(self.W,axis=0)  # vector normalization of thte weights
    def compute_objective_function(self, wordId, contextId, negativeIds):
        x_word = self.W[wordId]
        y_context = self.C[contextId]
        Z = self.C[negativeIds]
        obj = np.sum(np.log(sigmoid(x_word,y_context))) + np.sum(np.log(sigmoid(-x_word,Z)))
        return obj

    def trainWord(self, wordId, contextId, negativeIds):
        #use SGD to update weight matrices
        x_word = self.W[wordId]
        y_context = self.C[contextId]
        Z = self.C[negativeIds]
        grad_z_w = np.sum(sigmoid(x_word,Z)*Z,axis=0)
        grad_z_c = - sigmoid(x_word,Z)*x_word
        self.C[negativeIds] += self.lr*grad_z_c
        gradient_x = sigmoid(-x_word, y_context)*y_context - grad_z_w
        gradient_y = sigmoid(-x_word, y_context)*x_word
        self.W[wordId] = self.W[wordId] + self.lr*gradient_x
        self.C[contextId] = self.C[contextId] + self.lr*gradient_y

    def save(self,path):
        with open(path, 'wb') as f:
            pickle.dump(self.W, f)

    def similarity(self,word1,word2,W):
        mean_vector = W.mean(0)
        if word1 not in self.vocab and word2 not in self.vocab:
            return 0
        elif word1 not in self.vocab:
            v_1 = mean_vector
            v_2 = W[self.w2id[word2]]
            return v_1.dot(v_2) / (np.linalg.norm(v_1) * np.linalg.norm(v_2))
        elif word2 not in self.vocab:
            v_1 = W[self.w2id[word1]]
            v_2 = mean_vector
            return v_1.dot(v_2) / (np.linalg.norm(v_1) * np.linalg.norm(v_2))
        else:
			      #get the indexes
            v_1 = W[self.w2id[word1]]
            v_2 = W[self.w2id[word2]]

			      #cosine similarity
            cosine_sim = v_1.dot(v_2) / (np.linalg.norm(v_1) * np.linalg.norm(v_2))

            return cosine_sim

    def predict_similar_words(self, word, W, w2id, vocab, n_similar=5):
        """Predicts the n_similar most similar words to word"""
        key = lambda w: self.similarity(word,w,W,w2id,vocab) if w != word else - np.inf 
        similar_words = sorted(list(self.vocab.keys()), key=key, reverse=True)
        return similar_words[:n_similar]


    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            W = pickle.load(f)
        return W



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--text', help='path containing training data', required=True)
  parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
  parser.add_argument('--test', help='enters test mode', action='store_true')

  opts = parser.parse_args()

  sentences = text2sentences(opts.text)

  if not opts.test:
    sg = SkipGram(sentences, sub_sampling=True)
    sg.train()
    sg.save(opts.model)

  else:
    pairs = loadPairs(opts.text)
    sg = SkipGram(sentences)
    W = SkipGram.load(opts.model)
    for a,b,_ in pairs:
      # make sure this does not raise any exception, even if a or b are not in sg.vocab
      print(sg.similarity(a,b,W))
