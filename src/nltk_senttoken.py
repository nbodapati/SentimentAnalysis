from __future__ import division
import os
import numpy as np
from collections import defaultdict,Counter
from itertools import islice
import re
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

#sklearn decomposition packages
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD

import nltk
import copy
import pickle   
from nltk import word_tokenize
import time
#gensim word vector
from gensim.models import word2vec

#sklearn classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC 
from sklearn.cluster import KMeans
#tokenizer
from nltk.tokenize import RegexpTokenizer

pos_tag=1;
neg_tag=0;

#load_contexts functions
def load_word2vec(filename):
    # Returns a dict containing a {word: numpy array for a dense word vector} mapping.
    # It loads everything into memory.
    
    w2vec={}
    with open(filename,"r") as f_in:
        for line in f_in:
            line_split=line.replace("\n","").split()
            w=line_split[0]
            vec=np.array([float(x) for x in line_split[1:]])
            w2vec[w]=vec
    return w2vec

#load_contexts functions
def load_word2vec_from_vocab(filename,vocab):
    # Returns a dict containing a {word: numpy array for a dense word vector} mapping.
    # It loads everything into memory.
    
    w2vec={}
    with open(filename,"r") as f_in:
        for line in f_in:
            line_split=line.replace("\n","").split()
            w=line_split[0]
            if(w in list(vocab)):
              vec=np.array([float(x) for x in line_split[1:]])
              w2vec[w]=vec
    print("+++++Finished loading word2vec...") 
    return w2vec


def load_contexts(filename):
    # Returns a dict containing a {word: contextcount} mapping.
    # It loads everything into memory.

    data = {}
    for word,ccdict in stream_contexts(filename):
        data[word] = ccdict
    print "file %s has contexts for %s words" % (filename, len(data))
    return data

def stream_contexts(filename):
    # Streams through (word, countextcount) pairs.
    # Does NOT load everything at once.
    # This is a Python generator, not a normal function.
    for line in open(filename):
        word, n, ccdict = line.split("\t")
        n = int(n);
        ccdict = json.loads(ccdict);
        yield word, ccdict

def dom_polarity(sent_list,pos_emotion,neg_emotion,tokenizer):
    labels=[];
    for sent in sent_list:
        neg_count=0;pos_count=0;
        tokens=tokenizer.tokenize(sent);
        for t in tokens:
            if t in pos_emotion:
               pos_count+=1;
            elif t in neg_emotion: 
               neg_count+=1;
        if pos_count>=neg_count:
           labels.append(1);
        else:
           labels.append(0);        
               
    y_train=np.array([pos_tag]*5000+[neg_tag]*5000)
    print("Dominant polarity accuracy: ",np.mean(y_train==labels));     


class review_polarity:
    def __init__(self,split=0.5,reload=False,shuffle=True,remove=True,tokenize=False,wordvec=False,wordvec_file=None,
                       unigrams=False,bigrams_also=False,
                       only_bigrams=False,most_common_unigrams=False,pres_or_abs=False,dominant_polarity=False):

        self.vocab=set();
        self.skipgram_vocab=defaultdict(float);
        self.reload=reload
        self.x_train=None;
        self.y_train=None;
        self.x_val=None;
        self.y_val=None;
        self.x_test=None;
        self.y_test=None;

        self.split=split;
        self.shuffle=shuffle;
        self.remove=remove;
        self.tokenize=tokenize;
        self.unigrams=unigrams;
        self.bigrams_also=bigrams_also;
        self.only_bigrams=only_bigrams;
        self.most_common_unigrams=most_common_unigrams;
        self.pres_or_abs=pres_or_abs;
        self.word_vec=wordvec;
        self.wordvec_file=wordvec_file;
        self.model=None
        self.dominant_polarity=dominant_polarity;
        self.pos_emotion=['love','good','great','fantastic','wonderful','awesome','amazing',
                          'funny','comedy','interesting','special','happy','impressive','hilarious','excellent',
                          'brilliant','amusing','realistic','believable','terrific','enjoyed','exceptional',
                          'phenomenal','fascinating','intriguing','interesting','superb','witty','humorous',
                          'appealing','spectacular','wonderful','wonderfully','compelling','engaging',
                          'astounding','awesome','classic','delightful'];
        self.neg_emotion=['worst','bad','waste','poor','boring','stupid','worse','predictable','pathetic',
                          'terrible','expected','hate','awful','average','lame','bizarre','baffling','confusing'
                          'crap','pain','unfunny','disappointing','shallow','erratic','bewilderment','awful','unwatchable',
                          'horrible','irrelevant','painfully','failed','lackluster','sophorific'];

        #Regex tokenizer within the class. 
        self.tokenizer = RegexpTokenizer(r'\w+')
    

        self.pos_filepath="C:\\Users\\bodap\\OneDrive\\Documents\\Projects_Sem2\\NLP\\DataSets\\rotten_imdb"
        self.neg_filepath="C:\\Users\\bodap\\OneDrive\\Documents\\Projects_Sem2\\NLP\\DataSets\\rotten_imdb"

        self.pos_list=self.pos_filepath+"\\plot.tok.gt9.5000"; #List of files
        self.neg_list=self.neg_filepath+"\\quote.tok.gt9.5000";#List of files
        

        self.sent_list=[];
        sent_list=[];
        files_list=[self.pos_list,self.neg_list];
        		
        for l,file in enumerate(files_list):
            with open(file,'rb') as f:
                 sent_list.append(f.readlines());

        self.plot_list=sent_list[0];
        self.quote_list=sent_list[1];
        self.sent_list=self.plot_list+self.quote_list;

        print("Num sent in plot:",len(self.plot_list));
        print("Num sent in plot:",len(self.quote_list));
        print("Num sent in sent:",len(self.sent_list));
        if self.dominant_polarity:
            dom_polarity(self.sent_list,self.pos_emotion,self.neg_emotion,self.tokenizer);        
        

        if self.word_vec:
            if wordvec_file==None:
                tokens=[];
                for sent in self.sent_list:
                    toks=self.tokenizer.tokenize(sent);
                    T=[];
                    for t in toks:
                        try:
                           t=t.encode('utf-8').decode('ascii')
                           T.append(t)
                        except:
                            pass
                            #print("Skipping this special word from encapsulation: ",t);
                    #toks=[t.encode('utf-8') for t in toks];
                    tokens.append(T);
                model=word2vec.Word2Vec(tokens);
                model.save_word2vec_format("./sent_word2vec");
                return;
            else:
                wordvec_file=self.wordvec_file;

            self.build_vocab(); #this populates self.vocab
            print("REading from this: ",wordvec_file);
            word_to_vec_dict = load_word2vec_from_vocab(wordvec_file,self.vocab)
            flength=word_to_vec_dict['dog'].shape[0];
            self.x_train=np.zeros((len(self.sent_list),2*flength));
            num_sample=0;
            print("Word2vec sentence averaging,dot:");
            sent_except=0;
            for sent in self.sent_list:
                try:
                   sentences=nltk.sent_tokenize(sent.decode('utf-8'));
                except:
                    sent_except+=1;
                    pass 

                sentence_avg_vector=0.0;
                sentence_dot_vector=[1]*word_to_vec_dict['dog'].shape[0];
                #print sentences
                for sentence in sentences:
                    #print sentence
                    tokens_in_sentence=self.tokenizer.tokenize(sentence);
                    for token in tokens_in_sentence:
                        num_valid_tokens=0;
                        #print token
                        if(token in word_to_vec_dict.keys()):
                           num_valid_tokens+=1.0;
                           try:
                              sentence_avg_vector+=word_to_vec_dict[token];
                              sentence_dot_vector=np.dot(sentence_dot_vector,word_to_vec_dict[token]);
                           except:
                               pass
                    sentence_avg_vector/=(num_valid_tokens+0.1); 
                     
                self.x_train[num_sample,:flength]+=sentence_avg_vector; #Sentence average '
                self.x_train[num_sample,flength:]+=sentence_dot_vector;
                num_sample+=1;  
            print("Num sent excep:",sent_except)
            self.y_train=np.array([pos_tag]*len(self.plot_list)+[neg_tag]*len(self.quote_list)) 
            self.build_val_test_sets();     
            return;

        if self.reload:
           self.pipeline(); 
      


    def pipeline(self):
        print("building train and test sets..")
        if self.unigrams: 
            self.build_vocab()
        #Build full vocabulary and extract only adjectives from the loadset. 
            if self.remove:
               self.shrink_vocab()           
            self.build_train_set() 
            self.build_val_test_sets()

        elif self.only_bigrams:
            self.bigramFeatureExtractor(); #build the vocab,both trian and test set.
            self.build_train_set() 
            self.build_val_test_sets()

        elif self.bigrams_also:
            print("YEt to be done");     

    
    def bigramFeatureExtractor(self):
        #extract 4 skipgram word sets and make a dictionary out of them.
        for num_sent,sent in enumerate(self.sent_list):
            #print("Processing: ",num_sent);
            #this removes all the punct symbols except alphanumeric characters.
            tokens=self.tokenizer.tokenize(sent);    
                       
            #Create your bigrams/ngrams/skigrams.
            #ngs = nltk.skipgrams(tokens,2,5);
            ngs = nltk.ngrams(tokens,2);           
            #compute frequency distribution for all the bigrams in the text
            fdist = nltk.FreqDist(ngs)
            
            #build_skipgram.vocab
            for k,v in fdist.items():
                self.skipgram_vocab[k]+=v;
        #only retain those above a certain frequency
        temp_vocab=defaultdict(float);
        
        for k,v in self.skipgram_vocab.items():
            #print k,v
            tags=nltk.pos_tag(list(k));
            #if(tags[0][1]==tags[1][1] and tags[0][1]=='JJ'):
            l=[tag for w,tag in tags if tag=='JJ' or tag=='RB' or tag=='VBG' or tag=='VBN'] 
            #if l!=[]:
            if v>=5 and l!=[]:
               #print k,v,tags
               temp_vocab[k]+=v;      

        #re-reference skipgram_vocab to temp_vocab.                 
        self.skipgram_vocab=temp_vocab;
        print("Size of skipgram_vocab:",len(self.skipgram_vocab.keys()));

                 
    def build_vocab(self):
        for l in [self.plot_list,self.quote_list]:
            for sent in l:
                tokens=self.tokenizer.tokenize(sent);   
                c=Counter(tokens);
                self.vocab.update(c.keys());
       
        print("Length of vocab: ",len(self.vocab))
   
    def shrink_vocab(self):
        #REtain only adjectives.
        def extract_adjectives(tup):
            if tup[1]=='JJ':
                return tup[0]

        word_tags=nltk.pos_tag(self.vocab); #results in a list of tuples
        self.vocab=set(filter(extract_adjectives,word_tags));
        print("new vocab: ",self.vocab)     
        print("Length of Vocab with only adjectives: ",len(self.vocab));
   

    def build_train_set(self):
        #Builds x_train and y_train on full sent_list.
        if self.unigrams:
           self.x_train=np.zeros((len(self.sent_list),len(self.vocab)));
           vocab=self.vocab;
        elif self.only_bigrams:
            self.x_train=np.zeros((len(self.sent_list),len(self.skipgram_vocab)));
            vocab=self.skipgram_vocab;
        else:
            pass
             
        num_sample=0;
        for sent in self.sent_list:         
            tokens=self.tokenizer.tokenize(sent); 
            if self.unigrams:  
               c=Counter(tokens);
         
               for i,feature in enumerate(vocab):
                   self.x_train[num_sample,i]=c[feature];
               """
               if(sum(self.x_train[num_sample])==0):
                  print("Sample all zero*********************")
               print(nltk.pos_tag(self.sent_list[num_sample].split()))
               """    
               num_sample+=1;
               if(num_sample%50==0):
                  print("Extracted train set for num_samples: ",num_sample)
            else: 
               ngs = nltk.ngrams(tokens,2); 
               c=Counter(ngs);              
               for i,feature in enumerate(vocab):
                   self.x_train[num_sample,i]=c[feature]; 
               num_sample+=1;
               if(num_sample%50==0):
                  print("Extracted train set for num_samples: ",num_sample) 

        if self.pres_or_abs:
           self.x_train[self.x_train>0]=1;
       
        self.y_train=np.array([pos_tag]*len(self.plot_list)+[neg_tag]*len(self.quote_list))
                  

    def build_val_test_sets(self):
        shuffled_list=range(len(self.sent_list))
 
        if self.shuffle:
           np.random.shuffle(shuffled_list)
        split_point=np.floor(self.split*len(self.sent_list));
        split_point=7000;

        x_train=copy.copy(self.x_train);


        self.x_train=x_train[shuffled_list[0:split_point]]
        x_val=x_train[shuffled_list[split_point:]]
        self.x_val=x_val[0:np.floor(x_val.shape[0]/2)]
        self.x_test=x_val[np.floor(x_val.shape[0]/2):]

        y_train=copy.copy(self.y_train)
        self.y_train=y_train[shuffled_list[0:split_point]];
        y_val=y_train[shuffled_list[split_point:]]
        self.y_val=y_val[0:np.floor(y_val.shape[0]/2)]
        self.y_test=y_val[np.floor(y_val.shape[0]/2):]

        print("x_train shape:",self.x_train.shape)
        print("x_val shape:",self.x_val.shape)
        print("x_test shape:",self.x_test.shape)
        print("y_train shape:",self.y_train.shape)
        print("y_val shape:",self.y_val.shape)
        print("y_test shape:",self.y_test.shape)

    def load_datasets(self):
        start=time.time()
        data=pickle.load(open('sent_dataset.p','rb'))  
        print("Time for loading from pickle: ",time.time() -start)
        self.x_train=data['x_train']
        self.y_train=data['y_train']
        self.x_val=data['x_val']
        self.y_val=data['y_val']
        self.x_test=data['x_test']
        self.y_test=data['y_test']
        print("Done with loading datasets...")

    def get_datasets(self):
        
        if self.reload==False and self.word_vec!=True:
            print("Loading datasets:")
            self.load_datasets();

        data={}
        data['x_train']=self.x_train
        data['y_train']=self.y_train
        data['x_val']=self.x_val
        data['y_val']=self.y_val
        data['x_test']=self.x_test
        data['y_test']=self.y_test  
        
        if self.reload:
           """
           start=time.time() 
           pickle.dump(data,open('sent_dataset.p','wb'))
           print("Time for dump to pickle: ",time.time() - start)
           """
           self.reload=False
           
        return data
    
        
class classifiers:
    def __init__(self,data,length_norm=True,sqrt_norm=False,scale=False):
        self.knn=KNeighborsClassifier(n_neighbors=1, weights='uniform',p=2, metric='minkowski')
        self.dt=DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1);
        self.linearsvc=LinearSVC(penalty='l2', loss='squared_hinge', tol=0.0001, C=1.0)
        self.randomforest=RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                                                 min_samples_leaf=1)
        self.adaboost=AdaBoostClassifier(base_estimator=None,n_estimators=50,learning_rate=1.0)
        self.bagging=BaggingClassifier(base_estimator=None,n_estimators=10,max_features=1.0,bootstrap=True,bootstrap_features=True,
                                       random_state=0)
        self.nb=GaussianNB()
        self.multinomial_nb=MultinomialNB(alpha=1.0)
        self.lr=LogisticRegression();
        self.linearsvm=LinearSVC();
        self.polysvm=SVC();

        #preprocessing parameters
        self.length_normalize=length_norm
        self.scale=scale
        self.sqrt_normalize=sqrt_norm
        eps=0.1

        if self.length_normalize:
            x_train=data['x_train']
            x_val=data['x_val']
            x_test=data['x_test']
            
            data['x_train']=x_train/(np.sum(x_train**2,axis=1,keepdims=True)+eps)
            data['x_val']=x_val/(np.sum(x_val**2,axis=1,keepdims=True)+eps)
            data['x_test']=x_test/(np.sum(x_test**2,axis=1,keepdims=True)+eps)

        if self.sqrt_normalize:
           data['x_train']=data['x_train']/(np.sqrt(data['x_train'])+eps)
           data['x_val']=data['x_val']/(np.sqrt(data['x_val'])+eps)
           data['x_test']=data['x_test']/(np.sqrt(data['x_test'])+eps)
             

        if self.scale:
            x_train=data['x_train']
            x_val=data['x_val']
            x_test=data['x_test']
            
            train_mean=np.mean(x_train,axis=0,keepdims=True)
            val_mean=np.mean(x_val,axis=0,keepdims=True)
            test_mean=np.mean(x_test,axis=0,keepdims=True)
            train_std=np.std(x_train,axis=0,keepdims=True)
            val_std=np.std(x_val,axis=0,keepdims=True)
            test_std=np.std(x_test,axis=0,keepdims=True)

            data['x_train']=(x_train-train_mean)/(train_std+eps)
            data['x_val']=(x_val-val_mean)/(val_std+eps)
            data['x_test']=(x_test-test_mean)/(test_std+eps)

    
        self.start_training(data)

    def start_training(self,data):
        self.train_clustering(data)   
        self.train_template_matching(data)
        self.train_nb(data)
        self.train_ada(data)
        self.train_bagging(data)
        self.train_rf(data)
        self.train_knn(data)
        self.train_dt(data)
        self.train_lr(data)
        self.train_linearsvm(data)
        self.polysvm(data)

    def train_linearsvm(self,data):
        print("Training Linear SVM....")
        x_train=data['x_train']                                                
        y_train=data['y_train']
        x_val=data['x_val']
        y_val=data['y_val']
        x_test=data['x_test']
        y_test=data['y_test']

        #knn=copy.deepcopy(self.knn)
        knn=self.linearsvm #default knn.
        knn.fit(x_train,y_train)
        y_pred=knn.predict(x_train)
       
        print("LinearSVM training accuracy: ",np.mean(y_pred==y_train))
        y_pred=knn.predict(x_val)
        print("LinearSVM val accuracy: ",np.mean(y_pred==y_val))
        y_pred=knn.predict(x_test)
        print("LinearSVM test accuracy: ",np.mean(y_pred==y_test))

        #Tune on hyperparameters --coarse to fine search --for C or regularization strength.
        best_svm=None
        best_c=0
        val_acc=0.0
        for c_ in [0.01,0.1,1.0,10.0]:
            knn=LinearSVC(C=c_)
            knn.fit(x_train,y_train)

            y_pred=knn.predict(x_train)
            acc=np.mean(y_train==y_pred)
            print("Lsvm{0} train Val acc:{1} ".format(c_,acc))
            y_pred=knn.predict(x_val)
            acc=np.mean(y_val==y_pred)
            print("Lsvm{0} Val acc:{1} ".format(c_,acc))
            if val_acc <=acc:
                val_acc=acc
                best_lr=knn
                best_c=c_
        
        y_pred=best_lr.predict(x_test)                   
        print("Best_c{0} test_acc{1}".format(best_c,np.mean(y_pred==y_test)))


    def train_polysvm(self,data):
        print("Training Poly SVM....")
        x_train=data['x_train']                                                
        y_train=data['y_train']
        x_val=data['x_val']
        y_val=data['y_val']
        x_test=data['x_test']
        y_test=data['y_test']

        #knn=copy.deepcopy(self.knn)
        knn=self.polysvm #default knn.
        knn.fit(x_train,y_train)
        y_pred=knn.predict(x_train)
       
        print("PolySVM training accuracy: ",np.mean(y_pred==y_train))
        y_pred=knn.predict(x_val)
        print("LinearSVM val accuracy: ",np.mean(y_pred==y_val))
        y_pred=knn.predict(x_test)
        print("LinearSVM test accuracy: ",np.mean(y_pred==y_test))

        #Tune on hyperparameters --coarse to fine search --for C or regularization strength.
        best_svm=None
        best_c=0
        val_acc=0.0
        for c_ in [0.01,0.1,1.0,10.0]:
            knn=LinearSVC(C=c_)
            knn.fit(x_train,y_train)

            y_pred=knn.predict(x_train)
            acc=np.mean(y_train==y_pred)
            print("Lsvm{0} train Val acc:{1} ".format(c_,acc))
            y_pred=knn.predict(x_val)
            acc=np.mean(y_val==y_pred)
            print("Lsvm{0} Val acc:{1} ".format(c_,acc))
            if val_acc <=acc:
                val_acc=acc
                best_lr=knn
                best_c=c_
        
        y_pred=best_lr.predict(x_test)                   
        print("Best_c{0} test_acc{1}".format(best_c,np.mean(y_pred==y_test)))

    def train_nb(self,data):
        print("Training Gaussian Naive Bayes....")
        x_train=data['x_train']                                                
        y_train=data['y_train']
        x_val=data['x_val']
        y_val=data['y_val']
        x_test=data['x_test']
        y_test=data['y_test']

        #knn=copy.deepcopy(self.knn)
        knn=self.nb #default knn.
        knn.fit(x_train,y_train)
        y_pred=knn.predict(x_train)
       
        print("NB training accuracy: ",np.mean(y_pred==y_train))
        y_pred=knn.predict(x_val)
        print("NB val accuracy: ",np.mean(y_pred==y_val))
        y_pred=knn.predict(x_test)
        print("Nb test accuracy: ",np.mean(y_pred==y_test))
        """
        print("Training Multinomial Naive Bayes....")
        
        #knn=copy.deepcopy(self.knn)
        knn=self.multinomial_nb #default knn.
        knn.fit(x_train,y_train)
        y_pred=knn.predict(x_train)
       
        print("NB training accuracy: ",np.mean(y_pred==y_train))
        y_pred=knn.predict(x_val)
        print("NB val accuracy: ",np.mean(y_pred==y_val))
        y_pred=knn.predict(x_test)
        print("Nb test accuracy: ",np.mean(y_pred==y_test))


        #Tune on hyperparameters --coarse to fine search
        best_knn=None
        best_nn=0
        val_acc=0.0
        for alpha_ in [1.0,10.0,20.0]:
            knn=MultinomialNB(alpha=alpha_)
            knn.fit(x_train,y_train)

            y_pred=knn.predict(x_train)
            acc=np.mean(y_train==y_pred)
            print("M_NB{0} train Val acc:{1} ".format(alpha_,acc))
            y_pred=knn.predict(x_val)
            acc=np.mean(y_val==y_pred)
            print("MNB{0} Val acc:{1} ".format(alpha_,acc))
            if val_acc <=acc:
                acc=val_acc
                best_knn=knn
                best_nn=alpha_
        
        y_pred=best_knn.predict(x_test)                   
        print("Best_alpha{0} test_acc{1}".format(best_nn,np.mean(y_pred==y_test))) 
        """ 
    def train_lr(self,data):
        print("Training Gaussian Naive Bayes....")
        x_train=data['x_train']                                                
        y_train=data['y_train']
        x_val=data['x_val']
        y_val=data['y_val']
        x_test=data['x_test']
        y_test=data['y_test']

        #knn=copy.deepcopy(self.knn)
        knn=self.lr #default knn.
        knn.fit(x_train,y_train)
        y_pred=knn.predict(x_train)
       
        print("Logistic Regression/Max-entropy training accuracy: ",np.mean(y_pred==y_train))
        y_pred=knn.predict(x_val)
        print("LR val accuracy: ",np.mean(y_pred==y_val))
        y_pred=knn.predict(x_test)
        print("LR test accuracy: ",np.mean(y_pred==y_test))

        #Tune on hyperparameters --coarse to fine search --for C or regularization strength.
        best_lr=None
        best_c=0
        val_acc=0.0
        for c_ in [0.01,0.1,1.0,10.0]:
            knn=LogisticRegression(C=c_)
            knn.fit(x_train,y_train)

            y_pred=knn.predict(x_train)
            acc=np.mean(y_train==y_pred)
            print("LR{0} train Val acc:{1} ".format(c_,acc))
            y_pred=knn.predict(x_val)
            acc=np.mean(y_val==y_pred)
            print("LR{0} Val acc:{1} ".format(c_,acc))
            if val_acc <=acc:
                val_acc=acc
                best_lr=knn
                best_c=c_
        
        y_pred=best_lr.predict(x_test)                   
        print("Best_c{0} test_acc{1}".format(best_c,np.mean(y_pred==y_test)))     
                    
    def train_dt(self,data):
        print("Training DT...")
        x_train=data['x_train']                                                
        y_train=data['y_train']
        x_val=data['x_val']
        y_val=data['y_val']
        x_test=data['x_test']
        y_test=data['y_test']

        #knn=copy.deepcopy(self.knn)
        dt=self.dt #default knn.
        dt.fit(x_train,y_train)
        y_pred=dt.predict(x_train)
        print("DT training accuracy: ",np.mean(y_pred==y_train))
        y_pred=dt.predict(x_val)
        print("DT val accuracy: ",np.mean(y_pred==y_val))
        y_pred=dt.predict(x_test)
        print("DT test accuracy: ",np.mean(y_pred==y_test))
        #Tune on hyperparameters --coarse to fine search
        best_dt=None
        best_depth=0
        val_acc=0.0

        for depth in range(5,30,5):
            dt=DecisionTreeClassifier(criterion='gini',max_depth=depth)
            dt.fit(x_train,y_train)
            y_pred=dt.predict(x_val)
            acc=np.mean(y_val==y_pred)
            print("depth{0} Val acc:{1} ".format(depth,acc))
            if val_acc <=acc:
                val_acc=acc
                best_dt=dt
                best_depth=depth
        
        y_pred=best_dt.predict(x_test)                   
        print("Best_depth{0} test_acc{1}".format(best_depth,np.mean(y_pred==y_test)))          

    def train_template_matching(self,data):
       
        def cosine_similarity(v1,v2):
            #v1=v1.reshape(1,-1);
            #print(v1.shape,v2.shape);
            dot_pod=np.dot(list(v2),list(v1))
            v1_mod=np.linalg.norm(v1)
            v2_mod=np.linalg.norm(v2)
            return dot_pod/(v1_mod*v2_mod)

        def predict(test,centroid_pos,centroid_neg):
            pred_list=[]
            for v1 in range(test.shape[0]):                
                if cosine_similarity(test[v1],centroid_pos)>=cosine_similarity(test[v1],centroid_neg):
                   pred_list.append(pos_tag)
                else:
                    pred_list.append(neg_tag)
            return pred_list                

        print("Training for Template matching....")
        x_train=data['x_train']                                                
        y_train=data['y_train']
        x_val=data['x_val']
        y_val=data['y_val']
        x_test=data['x_test']
        y_test=data['y_test']

        centroid_pos=np.mean(x_train[y_train==pos_tag],axis=0,keepdims=True)
        centroid_neg=np.mean(x_train[y_train==neg_tag],axis=0,keepdims=True)
        print("Template matching train acc: ",np.mean(y_train==predict(x_train,centroid_pos,centroid_neg)))
        print("Template matching val acc: ",np.mean(y_val==predict(x_val,centroid_pos,centroid_neg)))
        print("Template matching test acc: ",np.mean(y_test==predict(x_test,centroid_pos,centroid_neg)))

    def train_clustering(self,data):
       
        def cosine_similarity(v1,v2):
            #v1=v1.reshape(1,-1);
            #print(v1.shape,v2.shape);
            dot_pod=np.dot(list(v2),list(v1))
            v1_mod=np.linalg.norm(v1)
            v2_mod=np.linalg.norm(v2)
            return dot_pod/(v1_mod*v2_mod)

        def predict(test,centroid_pos,centroid_neg):
            pred_list=[]
            for v1 in range(test.shape[0]):                
                if cosine_similarity(test[v1],centroid_pos)>=cosine_similarity(test[v1],centroid_neg):
                   pred_list.append(pos_tag)
                else:
                    pred_list.append(neg_tag)
            return pred_list                

        print("Training for Clustering kmeans,kmeans++ ....")
        x_train=data['x_train']                                                
        y_train=data['y_train']
        x_val=data['x_val']
        y_val=data['y_val']
        x_test=data['x_test']
        y_test=data['y_test']

        km=KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300);
        centres=km.cluster_centers_;

        centroid_pos=centres[0];
        centroid_neg=centres[1];
        print("clustering train acc: ",np.mean(y_train==predict(x_train,centroid_pos,centroid_neg)))
        print("clustering val acc: ",np.mean(y_val==predict(x_val,centroid_pos,centroid_neg)))
        print("clustering test acc: ",np.mean(y_test==predict(x_test,centroid_pos,centroid_neg)))

    def train_knn(self,data):
        print("Training KNN....")
        x_train=data['x_train']                                                
        y_train=data['y_train']
        x_val=data['x_val']
        y_val=data['y_val']
        x_test=data['x_test']
        y_test=data['y_test']

        #knn=copy.deepcopy(self.knn)
        knn=self.knn #default knn.
        knn.fit(x_train,y_train)
        y_pred=knn.predict(x_train)
        print(len(y_pred))
        print(len(y_train))
        print(y_pred==y_train)
        print("KNN training accuracy: ",np.mean(y_pred==y_train))
        y_pred=knn.predict(x_val)
        print("KNN val accuracy: ",np.mean(y_pred==y_val))
        y_pred=knn.predict(x_test)
        print("KNN test accuracy: ",np.mean(y_pred==y_test))

        #Tune on hyperparameters --coarse to fine search
        best_knn=None
        best_nn=0
        val_acc=0.0
        for nn in [4,5,6]:
            knn=KNeighborsClassifier(n_neighbors=nn, weights='uniform',p=2, metric='minkowski')
            knn.fit(x_train,y_train)

            y_pred=knn.predict(x_train)
            acc=np.mean(y_train==y_pred)
            print("NN{0} train Val acc:{1} ".format(nn,acc))
            y_pred=knn.predict(x_val)
            acc=np.mean(y_val==y_pred)
            print("NN{0} Val acc:{1} ".format(nn,acc))
            if val_acc <=acc:
                val_acc=acc
                best_knn=knn
                best_nn=nn
        
        y_pred=best_knn.predict(x_test)                   
        print("Best_nn{0} test_acc{1}".format(best_nn,np.mean(y_pred==y_test))) 

    def train_bagging(self,data):
        print("Training Bagging....")
        x_train=data['x_train']                                                
        y_train=data['y_train']
        x_val=data['x_val']
        y_val=data['y_val']
        x_test=data['x_test']
        y_test=data['y_test']

        ada=self.bagging
        ada.fit(x_train,y_train)
        y_pred=ada.predict(x_train)
        acc=np.mean(y_train==y_pred)
        print("Ada train Val acc:{0} ".format(acc))
        y_pred=ada.predict(x_val)
        acc=np.mean(y_val==y_pred)
        print("Ada Val acc:{0} ".format(acc))
        y_pred=ada.predict(x_test)
        acc=np.mean(y_test==y_pred)
        print("Ada test acc:{0} ".format(acc))

    def train_ada(self,data):
        print("Training Adaboost....")
        x_train=data['x_train']                                                
        y_train=data['y_train']
        x_val=data['x_val']
        y_val=data['y_val']
        x_test=data['x_test']
        y_test=data['y_test']

        ada=self.adaboost
        ada.fit(x_train,y_train)
        y_pred=ada.predict(x_train)
        acc=np.mean(y_train==y_pred)
        print("Ada train Val acc:{0} ".format(acc))
        y_pred=ada.predict(x_val)
        acc=np.mean(y_val==y_pred)
        print("Ada Val acc:{0} ".format(acc))
        y_pred=ada.predict(x_test)
        acc=np.mean(y_test==y_pred)
        print("Ada test acc:{0} ".format(acc))


    def train_rf(self,data):
        print("Training Random forest....")
        x_train=data['x_train']                                                
        y_train=data['y_train']
        x_val=data['x_val']
        y_val=data['y_val']
        x_test=data['x_test']
        y_test=data['y_test']

        ada=self.randomforest
        ada.fit(x_train,y_train)
        y_pred=ada.predict(x_train)
        acc=np.mean(y_train==y_pred)
        print("Ada train Val acc:{0} ".format(acc))
        y_pred=ada.predict(x_val)
        acc=np.mean(y_val==y_pred)
        print("Ada Val acc:{0} ".format(acc))
        y_pred=ada.predict(x_test)
        acc=np.mean(y_test==y_pred)
        print("Ada test acc:{0} ".format(acc))


class word_vectors:
    #word vectors from LDA,LSA,sparse context, dense context,word embeddings.
    #document as a sum of word_vectors
    def __init__(self,data,method='lsa'):
        print("Word vectors starts..")
        #self.lda_decomposition(data)
        if method=='lsa':
           self.lsa_decomposition(data)
        else:
            self.lda_decomposition(data)

    def lda_decomposition(self,data):
        print("LDA decomposition starts....")
        x_train=data['x_train']                                                
        y_train=data['y_train']
        x_val=data['x_val']
        y_val=data['y_val']
        x_test=data['x_test']
        y_test=data['y_test']
        lda=LatentDirichletAllocation()
        data['x_train']=lda.fit_transform(x_train,y_train)
        data['x_val']=lda.transform(x_val)
        data['x_test']=lda.transform(x_test)
        #print("x_train.shape= ",x_train.shape)

        print("data[x_train].shape= ",data['x_train'].shape)
        #print("First data sample after LDA: ",data['x_train'][0,:])
        print("Number all not zero?!! : ",data['x_train'].shape[0] - np.sum(np.sum(data['x_train'],axis=1,keepdims=True)==0)); 
 
    def lsa_decomposition(self,data):
        print("LSA/Truncated SVD decomposition starts....")
        x_train=data['x_train']                                                
        y_train=data['y_train']
        x_val=data['x_val']
        y_val=data['y_val']
        x_test=data['x_test']
        y_test=data['y_test']

        print("Number all not zero?!! : ",data['x_train'].shape[0] - np.sum(np.sum(data['x_train'],axis=1,keepdims=True)==0)); 

        lsa=TruncatedSVD(n_components=100)
        data['x_train']=lsa.fit_transform(x_train,y_train)
        data['x_val']=lsa.transform(x_val)
        data['x_test']=lsa.transform(x_test)
        #print("x_train.shape= ",x_train.shape)
        print("data[x_train].shape= ",data['x_train'].shape)
        #print("First data sample after LSA: ",data['x_train'][0,:])
        print("Number all not zero?!! : ",data['x_train'].shape[0] - np.sum(np.sum(data['x_train'],axis=1,keepdims=True)==0)); 
     

#if __name__=="__main__":
print("Creating sentiment token class");
#reload is like recompute teh feature vectors from scratch
#instead of loading from pickle.
"""
rp=review_polarity(dominant_polarity=True,unigrams=False,wordvec=True,wordvec_file=None,
                   only_bigrams=False,reload=True,remove=False); 
"""
 
rp=review_polarity(unigrams=False,wordvec=True,wordvec_file='./sent_word2vec',
                   only_bigrams=False,reload=True,remove=False,dominant_polarity=True); 
data=rp.get_datasets()
#clfs=classifiers(data)
#word_vec=word_vectors(data)

clfs=classifiers(data)



"""
print("***unigrams,reload=true,remove=false,pres_or_abs=false,LSA");
rp=review_polarity(unigrams=True,wordvec=False,wordvec_file='./glove.6B/glove.6B.50d.txt',
                   only_bigrams=False,reload=True,remove=False); 
data=rp.get_datasets()
#clfs=classifiers(data)
word_vec=word_vectors(data)
clfs=classifiers(data)

"""
"""
print("***unigrams,reload=true,remove=false,pres_or_abs=false,LDA");
rp=review_polarity(unigrams=True,wordvec=False,wordvec_file='./glove.6B/glove.6B.50d.txt',
                   only_bigrams=False,reload=True,remove=False); 
data=rp.get_datasets()
#clfs=classifiers(data)
word_vec=word_vectors(data,method='lda')
clfs=classifiers(data)


print("***unigrams,reload=false,remove=false,pres_or_abs=True,LSA");
rp=review_polarity(unigrams=True,wordvec=False,wordvec_file='./glove.6B/glove.6B.50d.txt',
                   only_bigrams=False,reload=True,remove=False,pres_or_abs=True); 
data=rp.get_datasets()
#clfs=classifiers(data)
word_vec=word_vectors(data)
clfs=classifiers(data)

print("***unigrams,reload=false,remove=false,pres_or_abs=True,LDA");
rp=review_polarity(unigrams=True,wordvec=False,wordvec_file='./glove.6B/glove.6B.50d.txt',
                   only_bigrams=False,reload=True,remove=False,pres_or_abs=True); 
data=rp.get_datasets()
#clfs=classifiers(data)
word_vec=word_vectors(data,method='lda')
clfs=classifiers(data)


print("***bigrams,reload=false,remove=false,pres_or_abs=True,LDA");
rp=review_polarity(unigrams=False,wordvec=False,wordvec_file='./glove.6B/glove.6B.50d.txt',
                   only_bigrams=True,reload=True,remove=False,pres_or_abs=True); 
data=rp.get_datasets()
#clfs=classifiers(data)
word_vec=word_vectors(data,method='lda')
clfs=classifiers(data)

"""
