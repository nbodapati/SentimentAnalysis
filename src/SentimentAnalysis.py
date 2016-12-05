#Load the data sets --subjectivity and IMDB.
#Learn of the statistics.
#Extract differnt types of features --> bag of words -counts,presence/not,unit leangth,square of feature, normalized,MinMaxscaler
#Feature extraction: PCA,bestSelector,selectKbest. 
#Train a basic Naive bayes or a log regression classifier and capture the results.
#Hyperparameter tuning: gridsearch,k-fold cross validation.

import numpy as np
#import sklearn 
from collections import defaultdict,Counter
import pickle
from itertools import islice
from sklearn.decomposition import PCA


filepath="C:\\Users\\bodap\\Dropbox\\NLP_Project\\DataSets\\rotten_imdb"
plot_file=filepath+"\\plot.tok.gt9.5000";
quote_file=filepath+"\\quote.tok.gt9.5000";


vocab=defaultdict(float);
sub_tag=1;
obj_tag=0;


plot_list=[];
quote_list=[];

files_list=[plot_file,quote_file];
sent_list=[];

for l,file in enumerate(files_list):
    with open(file,'rb') as f:
         sent_list.append(f.readlines());

plot_list=sent_list[0];
quote_list=sent_list[1];
print("Num sent in plot:",len(plot_list));
print("Num sent in plot:",len(quote_list));                 

#build vocabulary
for l in [plot_list,quote_list]:
    for sentences in plot_list:
        tokens=sentences.split();
        c=Counter(tokens);
        for key,occ in c.iteritems():
            vocab[key]+=occ;

#REmove most and least occuring words--limit the length of vocab to 5000.
sorted_vocab=sorted(vocab.items(),key=lambda x: -x[1]);   
#print(list(islice(vocab.iteritems(), 100)));
#print(sorted_vocab[:5000]);

remove_list=[]
for i in range(0,1):
    remove_list.append(sorted_vocab[i][0]);
    remove_list.append(sorted_vocab[-i][0]); #just take the keys.
    
vocab={key:vocab[key] for key in vocab.keys() if key not in remove_list}

#print(remove_list);
print(list(islice(vocab.iteritems(), 100)));

#print("Length of vocab: ",len(vocab.keys()));

total_sent=len(plot_list)+len(quote_list);
train_set=np.zeros((total_sent,len(vocab.keys())));
train_labels=np.zeros((1,total_sent));


num_sample=0;
for l in [plot_list,quote_list]:
    for sentences in l:
        tokens=sentences.split();
        c=Counter(tokens);
        #print(c);
        for num,key in enumerate(vocab.keys()):
            if(c[key]>0):
               print(num_sample,key,c[key]);
            train_set[num_sample,num]=c[key]; 
            #print(train_set[train_set>0]);
        num_sample+=1;

train_labels[0][0:len(plot_list)]=sub_tag;
train_labels[0][len(plot_list):total_sent]=obj_tag;


#Dimensionality reduction --svd?
"""
pca=PCA(n_components=0.5,whiten=False);

train_set=pca.fit_transform(train_set)
print("N_components post pca: ",pca.n_components_);
"""
#Break the dataset into train-test set 50%.
shuffled_list=range(0,total_sent);
np.random.shuffle(shuffled_list);
print("Shuffled list: ",shuffled_list);

partition=int(np.ceil(total_sent*0.7));

x_train=train_set[shuffled_list[0:partition]];
x_test=train_set[shuffled_list[partition:total_sent]];
y_train=train_labels[0][shuffled_list[0:partition]];
y_test=train_labels[0][shuffled_list[partition:total_sent]];

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

kbest=SelectKBest(score_func=f_regression,k=100);

x_train=kbest.fit_transform(x_train,y_train);
x_test=kbest.transform(x_test);


#Naive bayes classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train,y_train)
pred_train=clf.predict(x_train);
pred_test=clf.predict(x_test);
print("Accuracy NB on train set:",np.mean(pred_train==y_train));
print("Accuracy NB on test set:",np.mean(pred_test==y_test));

print("Few sent of test set: ",train_set[partition:partition+100]);
print("Our class tags: ",y_test[:100]);
print("Pred class tags: ",pred_test[:100]);


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(penalty='l2',C=1.0,multi_class='ovr');
clf.fit(x_train,y_train)
pred_train=clf.predict(x_train);
pred_test=clf.predict(x_test);
print("Accuracy on LR train set:",np.mean(pred_train==y_train));
print("Accuracy on LR test set:",np.mean(pred_test==y_test));
 



