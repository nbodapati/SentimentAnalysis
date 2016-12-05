# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 02:41:34 2016

@author: bodap
"""

#Load the data sets --subjectivity and IMDB.
#Learn of the statistics.
#Extract differnt types of features --> bag of words -counts,presence/not,unit leangth,square of feature, normalized,MinMaxscaler
#Feature extraction: PCA,bestSelector,selectKbest. 
#Train a basic Naive bayes or a log regression classifier and capture the results.
#Hyperparameter tuning: gridsearch,k-fold cross validation.
import os
import numpy as np
#import sklearn 
from collections import defaultdict,Counter
from itertools import islice
import re
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

def extract_unigram_features(sent_list,vocab,tokenize=False):
    total_sent=len(sent_list); 
    train_set=np.zeros((total_sent,len(vocab.keys())));
    num_sample=0;
    for sentences in sent_list:
        if tokenize==True:
            re.sub(r'[^A-Za-z0-9(),!?\'\`]', " ",sentences) 
        tokens=sentences.split();
        c=Counter(tokens);
        for num,key in enumerate(vocab.keys()):
            if(c[key]>0):
               print(num_sample,key,c[key]);
               train_set[num_sample,num]=c[key];
        num_sample+=1;
    return train_set
    
def extract_bigram_features(plot_list,quote_list,vocab):
    pass
    
def prepare_data(remove=False,tokenize=False,unigrams=True,bigrams_also=False,
                       only_bigrams=False,most_common_unigrams=False,pres_or_abs=False):
    pos_filepath="C:\\Users\\bodap\\OneDrive\\Documents\\Projects_Sem2\\NLP\\DataSets\\txt_sentoken\\pos\\"
    neg_filepath="C:\\Users\\bodap\\OneDrive\\Documents\\Projects_Sem2\\NLP\\DataSets\\txt_sentoken\\neg\\"
    vocab=defaultdict(float);
    pos_tag=1;
    neg_tag=0;

    pos_list=os.listdir(pos_filepath);
    neg_list=os.listdir(neg_filepath);

    sent_list=[]; 
    for _path,_list in [(pos_filepath,pos_list),(neg_filepath,neg_list)]:
        for l,file in enumerate(_list):
            with open(_path+file,'rb') as f:
                sent_list.append(f.read());

    print("Num sent in sent_list:",len(sent_list));                 

    #build vocabulary
    for sentences in sent_list:
        if tokenize==True:
           re.sub(r'[^A-Za-z0-9(),!?\'\`]', " ",sentences) 
        tokens=sentences.split();   
        c=Counter(tokens);
        for key,occ in c.iteritems():
            vocab[key]+=occ;
    #REmove 'n' most and least occuring words--limit the length of vocab to 5000.    
    #print(list(islice(vocab.iteritems(), 100)));
    #print(sorted_vocab[:5000]);
    if remove==True:
       sorted_vocab=sorted(vocab.items(),key=lambda x: -x[1]);    
       remove_list=[]
       for i in range(0,1000):
           remove_list.append(sorted_vocab[i][0]);
           remove_list.append(sorted_vocab[-i][0]); #just take the keys. 
       vocab={key:vocab[key] for key in vocab.keys() if key not in remove_list}
       print("Top 100 vocab items after remove list: ",list(islice(vocab.iteritems(), 100)));

    total_sent=len(sent_list); 
    train_labels=np.zeros((1,total_sent));
    if only_bigrams !=True:
       if unigrams==True:
          train_set=extract_unigram_features(sent_list,vocab);
       if bigrams_also==True:
           train_set=np.hstack((train_set,extract_bigram_features(sent_list,vocab)));
    else:
        train_set=extract_bigram_features(sent_list,vocab);
    
    if pres_or_abs==True:
        train_set[train_set>0]=1.0;      
        
    train_labels[0][0:len(pos_list)]=pos_tag;
    train_labels[0][len(pos_list):total_sent]=neg_tag;
    return (pos_list,neg_list,sent_list,vocab,train_set,train_labels)
    
 
def split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=False,normalize=False,scale=False):
    total_sent=train_set.shape[0];
    #Break the dataset into train-test set 50%.
    if normalize==True:
        #divide each document by its length/total number of words.
        doc_lengths=np.sum(train_set,axis=1,keepdims=True);
        train_set=train_set/doc_lengths;
    if scale==True:
        doc_mean=np.mean(train_set,axis=0,keepdims=True);
        doc_var=np.var(train_set,axis=0,keepdims=True);
        train_set=(train_set - doc_mean)/doc_var;
        
    if shuffle !=True:
       print("**********************************No shuffling*******************************************")
       x_train=np.vstack((train_set[0:len(plot_list)/2],train_set[len(plot_list):len(plot_list)+len(quote_list)/2]));
       x_test=np.vstack((train_set[len(plot_list)/2:len(plot_list)],train_set[len(plot_list)+len(quote_list)/2:]));
       y_train=np.hstack((train_labels[0][0:len(plot_list)/2],train_labels[0][len(plot_list):len(plot_list)+len(quote_list)/2]));
       y_test=np.hstack((train_labels[0][len(plot_list)/2:len(plot_list)],train_labels[0][len(plot_list)+len(quote_list)/2:]));
    else:
        print("***************************************Shuffled********************************************")
        shuffled_list=range(0,total_sent);
        np.random.shuffle(shuffled_list);
        #print("Shuffled list: ",shuffled_list);

        partition=int(np.ceil(total_sent*0.7));
        x_train=train_set[shuffled_list[0:partition]];
        x_test=train_set[shuffled_list[partition:total_sent]];
        y_train=train_labels[0][shuffled_list[0:partition]];
        y_test=train_labels[0][shuffled_list[partition:total_sent]];


    kbest=SelectKBest(score_func=f_regression,k=100);
    x_train=kbest.fit_transform(x_train,y_train);
    x_test=kbest.transform(x_test);
    return (x_train,x_test,y_train,y_test)
    

def classifiers(x_train,x_test,y_train,y_test):
    #Naive bayes classifier
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    pred_train=clf.predict(x_train);
    pred_test=clf.predict(x_test);
    print("Accuracy NB on train set:",np.mean(pred_train==y_train));
    print("Accuracy NB on test set:",np.mean(pred_test==y_test));

    from sklearn.linear_model import LogisticRegression
    clf=LogisticRegression(penalty='l2',C=1.0,multi_class='ovr');
    clf.fit(x_train,y_train)
    pred_train=clf.predict(x_train);
    pred_test=clf.predict(x_test);
    print("Accuracy on LR train set:",np.mean(pred_train==y_train));
    print("Accuracy on LR test set:",np.mean(pred_test==y_test));
    
"""    
plot_list,quote_list,sent_list,vocab,train_set,train_labels =prepare_data(tokenize=False)   
 
print("Plain classifiers")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=False,normalize=False,scale=False)
classifiers(x_train,x_test,y_train,y_test)
    
print("Shuffled data set")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=True,normalize=False,scale=False)
classifiers(x_train,x_test,y_train,y_test)

print("unshuffled,length normalized")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=False,normalize=True,scale=False)    
classifiers(x_train,x_test,y_train,y_test)

print("unshuffled,scaled")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=False,normalize=False,scale=True)  
classifiers(x_train,x_test,y_train,y_test)

print("shuffled,normalized")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=True,normalize=True,scale=False)   
classifiers(x_train,x_test,y_train,y_test)

print("shuffled,scaled");
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=True,normalize=False,scale=True)   
classifiers(x_train,x_test,y_train,y_test)
""" 

print("Special tokenize technique");
#pos_list,neg_list,sent_list,vocab,train_set,train_labels
plot_list,quote_list,sent_list,vocab,train_set,train_labels =prepare_data(tokenize=True)   
 
print("Plain classifiers")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=False,normalize=False,scale=False)
classifiers(x_train,x_test,y_train,y_test)
    
print("Shuffled data set")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=True,normalize=False,scale=False)
classifiers(x_train,x_test,y_train,y_test)

print("unshuffled,length normalized")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=False,normalize=True,scale=False)    
classifiers(x_train,x_test,y_train,y_test)

print("unshuffled,scaled")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=False,normalize=False,scale=True)  
classifiers(x_train,x_test,y_train,y_test)

print("shuffled,normalized")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=True,normalize=True,scale=False)   
classifiers(x_train,x_test,y_train,y_test)

print("shuffled,scaled");
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=True,normalize=False,scale=True)   
classifiers(x_train,x_test,y_train,y_test)



print("Presence or absence of the features");
plot_list,quote_list,sent_list,vocab,train_set,train_labels =prepare_data(tokenize=True,pres_or_abs=True)   
 
print("Plain classifiers")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=False,normalize=False,scale=False)
classifiers(x_train,x_test,y_train,y_test)
    
print("Shuffled data set")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=True,normalize=False,scale=False)
classifiers(x_train,x_test,y_train,y_test)

print("unshuffled,length normalized")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=False,normalize=True,scale=False)    
classifiers(x_train,x_test,y_train,y_test)

print("unshuffled,scaled")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=False,normalize=False,scale=True)  
classifiers(x_train,x_test,y_train,y_test)

print("shuffled,normalized")
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=True,normalize=True,scale=False)   
classifiers(x_train,x_test,y_train,y_test)

print("shuffled,scaled");
x_train,x_test,y_train,y_test =split_dataset(plot_list,quote_list,sent_list,train_set,train_labels,shuffle=True,normalize=False,scale=True)   
classifiers(x_train,x_test,y_train,y_test)

