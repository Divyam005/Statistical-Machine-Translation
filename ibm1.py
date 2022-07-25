from collections import defaultdict
import numpy as np
import pickle
from tqdm import tqdm
import os
class IBM_Model1():
    def __init__(self, iterations = 5):
        self.iterations = iterations

        english_file = open('corpus.en','r+')
        spanish_file = open('corpus.es','r+')

        self.eng_sent = english_file.readlines()
        self.esp_sent = spanish_file.readlines()

        self.num_sent = len(self.eng_sent)

        assert len(self.esp_sent) == self.num_sent

        english_file.close()
        spanish_file.close()

        pass
    
    def init_t(self):
        self.t = defaultdict(int)
        for k in range(self.num_sent):
            en_sent = ('_NULL_ '+self.eng_sent[k]).rstrip().split(' ')
            es_sent = self.esp_sent[k].rstrip().split(' ')
            for i in range(len(es_sent)):
                for j in range(len(en_sent)):
                    key = ' '.join([es_sent[i], en_sent[j]])
                    self.t[key] = 1.0/self.eng_word_dict[en_sent[j]]




    def train(self):

        self.eng_word_dict = defaultdict(int)
        for sent in self.eng_sent:
            sent = sent.rstrip().split(' ')
            self.eng_word_dict['_NULL_'] += 1
            for word in sent:
                self.eng_word_dict[word] += 1
        
        ## Translation Probabilities
        self.init_t()


        for it in range(self.iterations):
            count = defaultdict(int)
            for k in tqdm(range(self.num_sent)):
                en_sent = ('_NULL_ '+self.eng_sent[k]).rstrip().split(' ')
                es_sent = self.esp_sent[k].rstrip().split(' ')

                for i in range(len(es_sent)):
                    den = 0
                    for j in range(len(en_sent)):
                        den += self.t[' '.join([es_sent[i], en_sent[j]])]
                    
                    for j in range(len(en_sent)):
                        delta = self.t[' '.join([es_sent[i], en_sent[j]])]/den

                        count[' '.join([en_sent[j], es_sent[i]])] += delta 
                        count[en_sent[j]] += delta 
                        # count[' '.join([str(j),str(i),str(len(en_sent)),str(len(es_sent))])] += delta
                        # count[' '.join([str(i),str(len(en_sent)),str(len(es_sent))])] += delta 

            for k in range(self.num_sent):
                en_sent = ('_NULL_ '+self.eng_sent[k]).rstrip().split(' ')
                es_sent = self.esp_sent[k].rstrip().split(' ')

                for i in range(len(es_sent)): 
                    for j in range(len(en_sent)):
                        self.t[' '.join([es_sent[i], en_sent[j]])] = count[' '.join([en_sent[j], es_sent[i]])] / count[en_sent[j]]
            
            temp_predictions = model.predict(eng_dev_file='dev.en',esp_dev_file='dev.es')

            out_file = open('temp_file.p1.out','w+')
            for entry in temp_predictions:
                out_file.write('{} {} {}\n'.format(entry[0],entry[1],entry[2]))
            out_file.close()
            os.system('python eval_alignment.py dev.key temp_file.p1.out')
    
    def load_params(self,path):
        self.t = pickle.load(open(path,'rb'))
    
    def predict(self, eng_dev_file,esp_dev_file):
        eng_dev = open(eng_dev_file,'r+')
        esp_dev = open(esp_dev_file,'r+')

        eng_sent = eng_dev.readlines()
        esp_sent = esp_dev.readlines()
        
        prediction = []
        for k in range(len(eng_sent)):
            
            en_sent =  eng_sent[k].rstrip().split(' ')
            es_sent = esp_sent[k].rstrip().split(' ')

            for i in range(len(es_sent)): 
                pos = 0
                max_prob = 0
                for j in range(len(en_sent)):
                    if self.t[' '.join([es_sent[i], en_sent[j]])] > max_prob:
                        pos = j
                        max_prob = self.t[' '.join([es_sent[i], en_sent[j]])]
                prediction.append((k+1,pos+1,i+1))
        return prediction

if __name__ == "__main__": 
    model = IBM_Model1()
    model.train()
    pickle.dump(model.t, open('ibm1_iter5_final.pkl','wb'))
    model.load_params('ibm1_iter5_final.pkl')
    prediction = model.predict(eng_dev_file='dev.en',esp_dev_file='dev.es')

    out_file = open('alignment.p1.out','w+')
    for entry in prediction:
        out_file.write('{} {} {}\n'.format(entry[0],entry[1],entry[2]))
    out_file.close()