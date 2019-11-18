#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset=pd.read_csv("Mall_Customers.csv")
x=dataset.iloc[:,[3,4]].values

#import the Dendogram to find the optimal number of clustrers
import scipy.cluster.hierarchy as sch
dendogram=sch.dendogram(sch.linkage(x,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian Distance')
plt.legend()
plt.show()

#Fitting Hierarchial Clustering to the mall Dataset

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)

#Visualising the Clusters
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='Carefull')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='blue',label='Standard')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='green',label='Target')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='cyan',label='Careless')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='magenta',label='Sensible')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1=100)')
plt.legend()
plt.show()