from collections import defaultdict
from email.policy import default
import numpy as np
import pickle
from tqdm import tqdm
import os
class IBM_Model2():
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

        ## Translation Probabilities take from ibm1 model
        self.load_params('ibm1_iter5_final.pkl')

        pass
    
    def init_q(self):
        self.q = defaultdict(int)
        for k in range(self.num_sent):
            en_sent = ('_NULL_ '+self.eng_sent[k]).rstrip().split(' ')
            es_sent = self.esp_sent[k].rstrip().split(' ')
            for i in range(len(es_sent)):
                for j in range(len(en_sent)):
                    key = ' '.join( [str(j),str(i),str(len(en_sent)-1),str(len(es_sent))] )
                    self.q[key] = 1.0/len(en_sent)

    def train(self):

        self.eng_word_dict = defaultdict(int)
        for sent in self.eng_sent:
            sent = sent.rstrip().split(' ')
            self.eng_word_dict['_NULL_'] += 1
            for word in sent:
                self.eng_word_dict[word] += 1
        
        
        ### Init q(j|i,l,m)
        self.init_q()
        # print(self.t,self.q)

        for it in range(self.iterations):
            count = defaultdict(int)
            for k in tqdm(range(self.num_sent)):
                en_sent = ('_NULL_ '+self.eng_sent[k]).rstrip().split(' ')
                es_sent = self.esp_sent[k].rstrip().split(' ')

                for i in range(len(es_sent)):
                    den = 0
                    for j in range(len(en_sent)):
                        # print( self.t[' '.join([es_sent[i], en_sent[j]])])
                        den += self.t[' '.join([es_sent[i], en_sent[j]])] * self.q[' '.join([str(j),str(i),str(len(en_sent)-1),str(len(es_sent))])]
                    
                    for j in range(len(en_sent)):
                        delta = self.t[' '.join([es_sent[i], en_sent[j]])] * self.q[' '.join([str(j),str(i),str(len(en_sent)-1),str(len(es_sent))])]/den

                        count[' '.join([en_sent[j], es_sent[i]])] += delta 
                        count[en_sent[j]] += delta 
                        count[' '.join([str(j),str(i),str(len(en_sent)-1),str(len(es_sent))])] += delta
                        count[' '.join([str(i),str(len(en_sent)-1),str(len(es_sent))])] += delta 

            for k in range(self.num_sent):
                en_sent = ('_NULL_ '+self.eng_sent[k]).rstrip().split(' ')
                es_sent = self.esp_sent[k].rstrip().split(' ')

                for i in range(len(es_sent)): 
                    for j in range(len(en_sent)):
                        self.t[' '.join([es_sent[i], en_sent[j]])] = count[' '.join([en_sent[j], es_sent[i]])] / count[en_sent[j]]
                        key = ' '.join( [str(j),str(i),str(len(en_sent)-1),str(len(es_sent))] )
                        self.q[key] = count[' '.join([str(j),str(i),str(len(en_sent)-1),str(len(es_sent))])]/count[' '.join([str(i),str(len(en_sent)-1),str(len(es_sent))])]
            temp_predictions = model.predict(eng_dev_file='dev.en',esp_dev_file='dev.es')

            out_file = open('temp_file.p2.out','w+')
            for entry in temp_predictions:
                out_file.write('{} {} {}\n'.format(entry[0],entry[1],entry[2]))
            out_file.close()
            os.system('python eval_alignment.py dev.key temp_file.p2.out')
    
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
                    prob = self.t[' '.join([es_sent[i], en_sent[j]])] * self.q[' '.join([str(j+1),str(i),str(len(en_sent)),str(len(es_sent))])]
                    if prob > max_prob:
                        pos = j
                        max_prob = prob
                prediction.append((k+1,pos+1,i+1))
        return prediction

if __name__ == "__main__": 
    model = IBM_Model2()
    model.train()
    # pickle.dump(model.t, open('ibm2_params_t.pkl','wb'))
    # pickle.dump(model.q, open('ibm2_params_q.pkl','wb'))
    # model.load_params('ibm2_params.pkl')
    prediction = model.predict(eng_dev_file='dev.en',esp_dev_file='dev.es')

    out_file = open('alignment.p2.out','w+')
    for entry in prediction:
        out_file.write('{} {} {}\n'.format(entry[0],entry[1],entry[2]))
    out_file.close()