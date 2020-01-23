#Imputation of missing data with neural network for classification

#Algorithm 1

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.datasets
import visdom 
from matplotlib import pyplot as pt
from collections import defaultdict


#Iris data로 분석을 진행해보자
iris=sklearn.datasets.load_iris()
iris.keys()
x=iris.data
y=iris.target
#cross validation my self
Indice=list(range(len(y)))
np.random.seed(11)
np.random.shuffle(Indice)
indice=[]
for i in range(10):
    a=len(y)//10
    indice.append(Indice[a*i:a*(i+1)])
indice=np.array(indice)
np.random.choice(10,2,False)
#K mean clustering
#allocation update
def euclidean_dist(x,y):
    if type(x) or type(y) == list:
        x=np.array(x)
        y=np.array(y)
    return np.sqrt(np.sum((x-y)**2))

euclidean_dist([3,4],[0,0])
    
class k_mean:
    def __init__(self,k,input_data):
        # k : # of cluster
        # input data : m(# of data) * p(# of features)
        self.input_data=input_data
        self.k=k
        self.cluster={}
        indice=np.random.choice(len(self.input_data),self.k,False)
        for i in range(self.k):
            self.cluster[i]=self.input_data[indice[i]] #randomize
        
    def allocation(self):
        self.allocation_dict=dict()
        for i in range(self.k):
            self.allocation_dict[i]=[]
        for j in range(len(self.input_data)):
            self.allocation_dict[min([i for i in range(self.k)],key=lambda i : euclidean_dist(self.cluster[i],self.input_data[j]))].append(j)
        #return self.allocation
    def update(self):
        for i in range(self.k):
            if self.allocation_dict[i]: #data들이 있는 경우에만 update
                self.cluster[i]=np.mean(self.input_data[self.allocation_dict[i]],axis=0)
        return self.cluster
    
    def cost(self):
        #sum(rij(xi-cj)**2)
        s=0
        for i in range(self.k):
            s+=np.sum((self.input_data[self.allocation_dict[i]]-self.cluster[i])**2)
        return s
# k mean test
#vis.close(env='main')
########################################################################################
inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]
inputs=np.array(inputs)
model=k_mean(3,inputs)
model.allocation()
model.cost()
for epoch in range(10):
    model.update()
    model.allocation()
    print(model.cost())
