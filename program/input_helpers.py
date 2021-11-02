import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip
from random import random
from preprocess import MyVocabularyProcessor
from nltk.tag import StanfordPOSTagger
import sys
import os
import importlib
importlib.reload(sys)
#sys.setdefaultencoding("utf-8")

class InputHelper(object):

    def getTsvData(self, filepath, qc_label_list):
        print("Loading training data from "+filepath)
        x1 = []    #question
        x2 = []    #answer
        y1 = []    #best answer label
        y2 = []    #category of que
        # positive samples from file
        for line in open(filepath, encoding='utf-8'):
            l=line.strip().split("\t")
            if len(l) < 4:
                continue
            if l[3] not in qc_label_list.keys():
                continue
            x1.append(l[0].lower())
            x2.append(l[1].lower())
            y1.append(int(l[2]))
            y2.append(int(qc_label_list[l[3]]))
       
        return np.asarray(x1),np.asarray(x2),np.asarray(y1),np.asarray(y2)

    def getTsvTestData(self, filepath):
        print("Loading testing/labelled data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        # positive samples from file
        for line in open(filepath):
            l=line.strip().split("\t")
            if len(l)<3:
                continue
            x1.append(l[0].lower())
            x2.append(l[1].lower())
            y.append(int(l[2])) #np.array([0,1]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)


    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def dumpValidation(self,x1_text,x2_text,y,shuffled_index,dev_idx,i):
        print("dumping validation "+str(i))
        x1_shuffled=x1_text[shuffled_index]
        x2_shuffled=x2_text[shuffled_index]
        y_shuffled=y[shuffled_index]
        x1_dev=x1_shuffled[dev_idx:]
        x2_dev=x2_shuffled[dev_idx:]
        y_dev=y_shuffled[dev_idx:]
        del x1_shuffled
        del y_shuffled
        with open('validation.txt'+str(i),'w') as f:
            for text1,text2,label in zip(x1_dev,x2_dev,y_dev):
                f.write(str(label)+"\t"+text1+"\t"+text2+"\n")
            f.close()
        del x1_dev
        del y_dev

    # Data Preparatopn
    # ==================================================
    def char2id(self, alphabet):
        return {char:i+1 for i, char in enumerate(alphabet)}

    def getQClabel(self, label_file):
        qc_label_list = {}
        with open(label_file, 'r') as f:
            i = 0
            for line in f:
                l = line.strip()
                qc_label_list[l] = i
                i += 1
        return qc_label_list

    def loadMap(self, token2id_filepath):
        if not os.path.isfile(token2id_filepath):
            print("file not exist, building map")
            buildMap()

        token2id = {}
        id2token = {}
        with open(token2id_filepath, encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()#.decode("utf-8")
                token = row.split('\t')[0]
                token_id = int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token
        return token2id, id2token

    def saveMap(self, vocab):
        with open("char2id", "wb") as outfile:
            for idx in range(len(vocab)):
                outfile.write((vocab[idx] + "\t" + str(idx)  + "\r\n").encode(encoding="utf-8"))
        print("saved map betweein token and id")

    def getEmbeddings(self, infile_path, embedding_size):
        char2id, id_char = self.loadMap("char2id")
        row_index = 0
        emb_matrix = np.zeros((len(char2id.keys()), embedding_size))
        with open(infile_path, "r", encoding="utf-8") as infile:
            for row in infile:
                row = row.strip()
                row_index += 1
                items = row.split()
                char = items[0]
                emb_vec = [float(val) for val in items[1:]]
                if char in char2id:
                    emb_matrix[char2id[char]] = emb_vec
        return emb_matrix


    def getAdditionalFeature(self, q_set, a_set):
        with open('./trecQA/stop_words.txt','r') as infile:
            stopwords = set()
            for line in infile:
                stopwords.add(line.strip().lower())
        add_fea = []
        flatten = lambda l: [item for sublist in l for item in sublist.split()]
        q_vocab = list(set(flatten(q_set)))
        idf = {}
        for w in q_vocab:
            idf[w] = np.log(float(len(q_set)) / len([1 for s1 in q_set if w in s1]))
        #print idf
        for i in range(len(q_set)):
            s1 = q_set[i].lower().split()
            s2 = a_set[i].lower().split()
            word_cnt = len([word for word in s1 if word in s2])
            word_cnt_stop = len([word for word in s1 if (word not in stopwords) and (word in s2)])
            wgt_word_cnt = sum([idf[word] for word in s1 if word in s2])
            wgt_word_cnt_stop = sum([idf[word] for word in s1 if (word not in stopwords) and (word in s2)])
            add_fea.append([word_cnt,word_cnt_stop,wgt_word_cnt,wgt_word_cnt_stop])
        return np.asarray(add_fea)

    def getQCDataSets(self, vocab_processor, qc_text):
        qc_input = []
        qc_text_tokens = []
        for tokens, ids in vocab_processor.padSequence(qc_text, mode='qc'):
            qc_input.append(ids)
            qc_text_tokens.append(tokens)
        qc_input = np.asarray(list(qc_input))
        qc_text_tokens = np.asarray(list(qc_text_tokens))
        qc_input_char = np.asarray(list(vocab_processor.padChar(qc_text)))
        return qc_input, qc_input_char

    def getDataSets(self, training_paths, dev_paths, max_seq_len, max_word_len, batch_size, alphabet, label_file):
        sum_no_of_batches = 0
        qc_label_list = self.getQClabel(label_file)
        x1_text, x2_text, y1, y2 =self.getTsvData(training_paths,qc_label_list)
        add_fea = self.getAdditionalFeature(x1_text,x2_text)
        # Build vocabulary
        print("Building vocabulary")
        alphabet_id = self.char2id(alphabet)
        vocab_processor = MyVocabularyProcessor(max_seq_len, max_word_len, min_frequency=0, alphabet_id=alphabet_id)
        x_text = np.concatenate((x2_text,x1_text),axis=0)
        vocab_processor.fit_transform(x_text)
        vocab =  vocab_processor.vocabulary_.__dict__['_reverse_mapping']
        self.saveMap(vocab)
        print("Length of loaded vocabulary ={}".format(len(vocab_processor.vocabulary_)))

        que, que_char = self.getQCDataSets(vocab_processor, x1_text)
        ans, ans_char = self.getQCDataSets(vocab_processor, x2_text)


        '''
        label_tmp_list = ["location","human","entity","abbretiation","description","numerical"]
        label_id, _ = self.getQCDataSets(vocab_processor, label_tmp_list)
        qc_label_list['LOC']=label_id[0][0]
        qc_label_list['HUM']=label_id[1][0]
        qc_label_list['ENTY']=label_id[2][0]
        qc_label_list['ABBR']=label_id[3][0]
        qc_label_list['DESC']=label_id[4][0]
        qc_label_list['NUM']=label_id[5][0]
        #print(qc_label_list)
        #print(label_id)
        '''
        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y1)))
        que_shuffled = que[shuffle_indices]
        ans_shuffled = ans[shuffle_indices]
        que_char_shuffled = que_char[shuffle_indices]
        ans_char_shuffled = ans_char[shuffle_indices]
        y1_shuffled = y1[shuffle_indices]
        y2_shuffled = y2[shuffle_indices]
        add_fea_shuffled = add_fea[shuffle_indices]
        input_set = (que_shuffled, ans_shuffled, que_char_shuffled, ans_char_shuffled, y1_shuffled, y2_shuffled, add_fea_shuffled)  
        
        sum_no_of_batches = sum_no_of_batches+(len(y1)//batch_size)
        del x1_text
        del x2_text
        del vocab

        # Get dev data
        x1_text_dev, x2_text_dev, y1_dev, y2_dev =self.getTsvData(dev_paths,qc_label_list)
        add_fea_dev = self.getAdditionalFeature(x1_text_dev,x2_text_dev)
        que_dev, que_char_dev = self.getQCDataSets(vocab_processor, x1_text_dev)
        ans_dev, ans_char_dev = self.getQCDataSets(vocab_processor, x2_text_dev)
        input_set_dev = (que_dev, ans_dev, que_char_dev, ans_char_dev, y1_dev, y2_dev, add_fea_dev, x1_text_dev, x2_text_dev)

        return  input_set, input_set_dev, vocab_processor, sum_no_of_batches, qc_label_list


    def getTestDataSet(self, data_path, vocab_path, max_document_length):
        x1_temp,x2_temp,y = self.getTsvTestData(data_path)
        add_fea_test = self.getAdditionalFeature(x1_temp,x2_temp)

        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)

        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1,x2, y, x1_temp, x2_temp,add_fea_test
