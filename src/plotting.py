import numpy as np
import matplotlib.pyplot as plt

# Draws a bar chart specifically for Ques 3(a)
# Plots for each data set, 3 charts: training time, test time and test accuracy (got through kaggle)
# plots values of "knn" -> "decTree" -> "logReg" in order
def createBarChartForData(values,ylimit,dataSetName,yAxisName):
	#Create values and labels for bar chart
	inds = np.arange(3)
	labels = ["k-NN","DecTree","LogReg"]	

	#Plot a bar chart
	plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot
	plt.bar(inds, values, align='center') #This plots the data
	plt.grid(True) #Turn the grid on
	plt.ylabel(yAxisName) #Y-axis label
	plt.xlabel("Classification Method") #X-axis label
	plt.title("["+dataSetName+"]: "+yAxisName+" v/s Classifier") #Plot title
	plt.xlim(-0.5,3) #set x axis range
	plt.ylim(0,ylimit) #Set yaxis range

	#Set the bar labels
	plt.gca().set_xticks(inds) #label locations
	plt.gca().set_xticklabels(labels) #label values

	#Make sure labels and titles are inside plot area
	plt.tight_layout()

	#Save the chart
	plt.savefig("../Figures/DefaultClassifiers_"+dataSetName+"_"+yAxisName+"_BarChart.pdf")
	plt.gcf().clear()


# Draws a line graph specifically for Ques 5(a)
# Plots for 'Digits' data set, 6 charts: k v/s error when the other hyper-parameter values are fixed. (and k ranges from 1 to 4)
#	(a) uniform weight and p=1
#	(b) uniform weight and p=2
#	(c) uniform weight and p=3
#	(d) distance weight and p=1
#	(e) distance weight and p=2
#	(f) distance weight and p=3
# since accuracy values are passed, we subtract by "1" and then plot the results.
def createLineGraphForData(values,ylimit,titlePart):
	#Create values and labels for line graphs
	inds   =np.arange(1,5)
	labels =["Validation Error","Training Error"]

	#Plot a line graph
	plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
	plt.plot(inds,1-values[0,:],'or-', linewidth=3) #Plot the first series in red with circle marker
	plt.plot(inds,1-values[1,:],'sb-', linewidth=3) #Plot the first series in blue with square marker

	#This plots the data
	plt.grid(True) #Turn the grid on
	plt.ylabel("Error Value") #Y-axis label
	plt.xlabel("K (no. of neighbors)") #X-axis label
	plt.title("[Digits]: Error vs k "+titlePart) #Plot title
	plt.xlim(-0.1,5.0) #set x axis range
	plt.ylim(0,ylimit) #Set yaxis range
	plt.legend(labels,loc="best")

	#Make sure labels and titles are inside plot area
	plt.tight_layout()

	#Save the chart
	plt.savefig("../Figures/HyperParameterSelection_Digits_"+titlePart+"_lineChart.pdf")
	plt.gcf().clear()
