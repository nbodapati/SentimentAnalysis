# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 12:08:15 2016

@author: bodap
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_bar_chart(vals1,vals2,vals3,vals4,vals5,vals6,dataSetName,yAxisName):
    N = 2
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15       # the width of the bars
    fig=plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    ax = fig.add_subplot(111)

    yvals=vals1
    rects1 = ax.bar(ind, yvals, width, color='r')
    zvals = vals2
    rects2 = ax.bar(ind+width, zvals, width, color='g')
    kvals = vals3
    rects3 = ax.bar(ind+width*2, kvals, width, color='b')
    kvals = vals4
    rects4 = ax.bar(ind+width*3, kvals, width, color='gold')
    kvals = vals5
    rects5 = ax.bar(ind+width*4, kvals, width, color='#e5e4e2')
    kvals = vals6
    rects6 = ax.bar(ind+width*5, kvals, width, color='#f5e4e2')
    
    plt.grid(True) #Turn the grid on
    plt.ylabel(yAxisName) #Y-axis label
    plt.xlabel("Classification Method") #X-axis label
    plt.title("["+dataSetName+"]: "+yAxisName+" v/s Classifier") #Plot title
    #plt.xlim(-0.5,3) #set x axis range
    #plt.ylim(0,ylimit) #Set yaxis range     
    
    ax.set_xticks(ind+2*width)
    ax.set_xticklabels( ('NaiveBayes', 'LogisticRegression') )
    ax.legend( (rects1[0], rects2[0], rects3[0],rects4[0],rects5[0],rects6[0]), ('unshuffled','shuffled', 'unshuffled,norm', 'unshuffled,scaled','shuffled,norm','shuffled,scaled') )
    """
    #To print the yvalues on top of the bars-works well with values>0
    def autolabel(rects):
        for rect in rects:
            h = float(rect.get_height()*100)
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                   ha='center', va='bottom')
    
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    """
    plt.show()
	
"""  
print("Bar plots for the three datasets with classifiers  NB,LR ") ;
print("Dataset -- subjectivity analysis");
dataSetName="Subjectivity Dataset"
yAxisName="Testing Accuracy with punctuations removed"
noshuffle=[0.772,0.844]
shuffle=[ 0.778,0.842]
noshuffle_norm=[0.836, 0.808]
noshuffle_scale=[0.761,0.839]
shuffle_norm=[ 0.811,0.814]
shuffle_scale=[0.808,0.852]
plot_bar_chart(noshuffle,shuffle,noshuffle_norm,noshuffle_scale,shuffle_norm,shuffle_scale,dataSetName,yAxisName)

dataSetName="Subjectivity Dataset"
yAxisName="Training Accuracy"
noshuffle=[0.784, 0.8511]
shuffle=[0.8128,0.8554]
noshuffle_norm=[0.8451,0.8100]
noshuffle_scale=[0.7742,0.8508]
shuffle_norm=[0.826, 0.8204]
shuffle_scale=[0.776, 0.857]
plot_bar_chart(noshuffle,shuffle,noshuffle_norm,noshuffle_scale,shuffle_norm,shuffle_scale,dataSetName,yAxisName)

"""

dataSetName="Review Polarity Dataset"
yAxisName="Testing Accuracy with punctuations removed"
noshuffle=[0.787, 0.811]
shuffle=[0.780,0.791]
noshuffle_norm=[0.764,0.614]
noshuffle_scale=[0.779, 0.801]
shuffle_norm=[0.783,0.468]
shuffle_scale=[0.783,0.771]
plot_bar_chart(noshuffle,shuffle,noshuffle_norm,noshuffle_scale,shuffle_norm,shuffle_scale,dataSetName,yAxisName)

"""
dataSetName="Review Polarity Dataset"
yAxisName="Training Accuracy"
noshuffle=[ 0.8449, 0.893]
shuffle=[0.8442, 0.88428]
noshuffle_norm=[0.8479, 0.64]
noshuffle_scale=[0.844, 0.886]
shuffle_norm=[0.828,0.5657]
shuffle_scale=[ 0.821, 0.858]
plot_bar_chart(noshuffle,shuffle,noshuffle_norm,noshuffle_scale,shuffle_norm,shuffle_scale,dataSetName,yAxisName)


print("Bar plots for the three datasets with classifiers  NB,LR ") ;
print("Dataset -- subjectivity analysis");
dataSetName="Subjectivity Dataset"
yAxisName="Testing Accuracy  with pres/abs"
noshuffle=[0.767,0.836]
shuffle=[ 0.789,0.835]
noshuffle_norm=[0.836, 0.812]
noshuffle_scale=[ 0.761,0.834]
shuffle_norm=[0.817,0.8143]
shuffle_scale=[0.814,0.855]
plot_bar_chart(noshuffle,shuffle,noshuffle_norm,noshuffle_scale,shuffle_norm,shuffle_scale,dataSetName,yAxisName)
"""

dataSetName="Review Polarity Dataset"
yAxisName="Testing Accuracy with pres/abs"
noshuffle=[0.751, 0.817]
shuffle=[0.723, 0.814]
noshuffle_norm=[0.736,0.780]
noshuffle_scale=[0.746, 0.810]
shuffle_norm=[0.733,0.823]
shuffle_scale=[0.788,0.816]
plot_bar_chart(noshuffle,shuffle,noshuffle_norm,noshuffle_scale,shuffle_norm,shuffle_scale,dataSetName,yAxisName)