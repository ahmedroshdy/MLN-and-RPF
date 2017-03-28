import dbmoon
from math import exp
import matplotlib.pyplot as plt 
import numpy as np
import random
import kmeans
from sklearn.cluster import KMeans

#def ecuDistance(x,y):
#    np.sqrt(sum((x - y) ** 2))

#def getRandomCentroids(data, k ):
#    data=random.sample(data,k)
#    data=np.array(data)
#    return data
data = dbmoon.dbmoon(50)

#data = kmeans.new_data()
#Wv=kmeans.kmeans()
#plt.plot(Wv[:, 0],Wv[:, 1],'bo')
#plt.plot(data[:,0],data[:,1],'ro')
#plt.show()
#calculate max diff bet patterns
def max_diff(data): 
    dmax=0
    for i in range (0,len(data)-1):
        if(np.sqrt(sum((data[i,:2]-data[i+1,:2])**2))>dmax):
            dmax=np.sqrt(sum((data[i,:2]-data[i+1,:2])**2));
    return dmax


#print Wv.shape
#centers=random.sample(data[:,:2],random_k)
#centers=np.array(centers)
#Wv=centers[:,:2]
#function to calculate net_square
def net_sq(data,Wv,random_k):
    net=[]
    for i in range (0,len(data)):
        temp_row=[]
        for j in range (0,random_k):
            temp= sum((data[i,:2]-Wv[j,:2])**2)
            temp_row.append(temp)
        temp_row=np.array(temp_row)
        if i==0:
            net=temp_row
        if i>0:
            net=np.vstack((net,temp_row))   
    return net


def errorFunction( target, outu):
    error = 0.5*sum((target-outu)**2)
    return error

#input  x1,x2 (2000*2)
input = data[:,:2]
#output (2000*1)
target = data[:,2]
target= np.reshape(target, (len(data), 1))

#plt.plot(Wv[:, 0],Wv[:, 1],'bo')
#plt.plot(data[:,0],data[:,1],'ro')
#plt.show()
eita1=0.1
eita2=0.1
eita3=0.1
##initialization

#clusters=kmeans.k_means(random_k)
#clusters=np.array(clusters)
#clusters= np.reshape(clusters,(random_k,1))
#print clusters[1,0][0]


#weights of hidden layer
#centers=kmeans.kmeans()
#centers=random.sample(data[:,:2],random_k)
#centers=np.array(centers)
#Wv=centers[:,:2]

# function to calculate output from hidden layer
def outv(data,sigmav,Wv,random_k):
    return np.exp((-net_sq(data,Wv,random_k))/(2*(sigmav**2)))


#random_k = random.randint(1,10)

#trial to find k-means
#for k in range(0,20):
#    for i in range (0,len(data)):
#        for j in range (0,random_k):
#         acc[j,i]=np.sqrt((data[i,:2]-centers[j,:2])**2)

#print centers[:,:2] 
          

#1-update centers(hidden layer weight) (eita1)
#2- update sigma (eita2) 
#3-update wights(output layer weights)  (eita3)
total_accuracy=[]
for k in range(1,5):
    print "when no. of centers equal " + str(k)
    #centers=random.sample(data[:,:2],k)
    #centers=np.array(centers)
    #Wv=centers[:,:2]
    #k-means
    kmean = KMeans(n_clusters=k, random_state=0).fit(data)
    Wv=kmean.cluster_centers_
    random_k  = k
    Wv=Wv[:,:2]
    
    #weights of output layer
    Wu = 2*np.random.random((random_k,1))-1

    #radii of hidden nodes 
    sigma = (max_diff(data)**2)/(np.sqrt(2*len(data)))  
    #print sigma
    #sigma = 10.0
    sigmav = np.array(sigma)
    sigmav=np.tile(sigmav, (len(data),random_k))
    for iter in range(100):
        vi=outv(data,sigmav,Wv,random_k)
        output=np.dot(vi,Wu)
        #updating rules 
        #delta Wu = -eita3 * diffE/diffWu
        deltau= (eita3/len(data))* np.dot(vi.T,(target-output))
        #delta Wv = -eita2 * diffE/diffnetv * input
        deltai= (eita2/len(data))* np.dot((target-output),Wu.T) * ((-np.sqrt(net_sq(data,Wv,random_k)))/(np.power(sigmav,2))) * outv(data,sigmav,Wv,random_k) 
        deltav= np.dot(deltai.T,input)
        #delta sigma
        deltas= (eita1/len(data))* np.dot((target-output),Wu.T) * (net_sq(data,Wv,random_k))/(np.power(sigmav,3)) * outv(data,sigmav,Wv,random_k) 

        Wu += deltau
        #Wv +=deltav
        sigmav += deltas
        vi=outv(data,sigmav,Wv,random_k)
        output=np.dot(vi,Wu)  
        avgapprox = np.round(output) 
        print avgapprox
        error=errorFunction(target,output)
        #avgapprox = np.round(error,2)
        print "Total error in iter no."  +str(iter) + " is equal " + str(error) 
    print "Total classification error at " + str(k) + " hidden nodes is equal " + str(error)
    accuracy = 0 
    for i in range(len(data)):
        if(avgapprox[i]==target[i]):
            accuracy=accuracy+1

    per_accuracy= (float(accuracy)/float(len(data)))*100.0
    print "When no of centers(hidden nodes) equal " + str(k) + " accuracy percentage is " + str(per_accuracy)
    total_accuracy.append(per_accuracy)
    
total= np.array(total_accuracy)
total=np.round(total)
print "This is accuracy in each k "  + str(total)    
min_k=1
max = total[0]
for i in range(0,4):
    if (total[i]>max):
        max=total[i]
        min_k=i+1
print "Min. no. of k(hidden nodes) give us best accuracy is " + str(min_k)         







#print net[0]
#sum((data[:,:2]-centers[0,:2])**2)
#plt.plot(data[:, 0],data[:, 1])
#plt.show()