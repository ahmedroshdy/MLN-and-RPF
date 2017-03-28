import numpy as np
from math import exp
import matplotlib.pyplot as plt 
import time

#training sets (closed problem) that means number of datasets already defined 8 combinations
#this problem is not linearly sepearable so we will use backprobagation (there is no solution using linear output node)
# assume bias is zero in linear output node


#inputs
print "Hello Engineer OmarAmin \n This is Problem 3 on Tutorial 3 \n To experiment with different number of hidden nodes and find out the optimal numbers of nodes press 1 \n To repeat this using linear output node press 2 \n"
input = input("Select : ")



#inputs and outputs of the 3-parity problem
x = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]) 
y = np.array([[0],[1],[1],[0],[1],[0],[0],[0]])
eita = 0.2



def sigmoid(z):
    return 1/(1+np.exp(-z))

def errorFunction( y, yHat):
    error = 0.5*sum((y-yHat)**2)
    return error


##testing of sigmoid 
#testInput = np.arange(-6,6,0.01)
#plt.plot(testInput, sigmoid(testInput), linewidth= 2)
#plt.grid(1)
#plt.show()




#INITIALIZATION
totalError = np.full((8, 1), 0.1)
InputLayerSize = 3 
HiddenLayerSize = 0
OutputLayerSize = 1 
avgapprox=np.full((1, 1), 0.1)
#np.array_equal(avg,np.full((1, 1), 0.00))

while avgapprox:
    HiddenLayerSize+=1  
    #INITIALIZE WEIGHTS 
    wL1 = 2*np.random.random((InputLayerSize,HiddenLayerSize)) - 1
    wL2 = 2*np.random.random((HiddenLayerSize,OutputLayerSize)) - 1
    for iter in range(100000):
        #forward pass
        z1= np.dot(x,wL1)
        a1= sigmoid(z1)
        z2= np.dot(a1,wL2)
        if input==1:
            a2=sigmoid(z2)
            #backpropagation
            #don't confuse with my sympols
            #sigma2 as delta2 diff Error with respect to netL2 and deltaL2 is diff. error with respect to WL2
            sigma2 = (y-a2) * (a2*(1-a2))
            deltaL2 = np.dot(a1.T,sigma2)
            #sigma1 as delta1 diff Error with respect to netL1 and deltaL1 is diff. error with respect to WL1
            sigma1 = np.dot(sigma2,wL2.T) * (a1*(1-a1))
            deltaL1 = np.dot(x.T,sigma1)
        else:
            a2=z2
            sigma2 =(y-a2)
            deltaL2 = np.dot(a1.T,sigma2)
            sigma1 = np.dot(sigma2,wL2.T) * (a1*(1-a1))
            deltaL1 = np.dot(x.T,sigma1)
        #updating el weights
        wL2 = wL2 + (eita * deltaL2)
        wL1 = wL1  + (eita* deltaL1)
     ##try to add momentum and get thewe result 
    #calculating output after finsihing training
    #a2= Forwardpass(x,wL1,wL2,input)
    z1= np.dot(x,wL1)
    a1= sigmoid(z1)
    z2= np.dot(a1,wL2)
    #a2=activationfn(z2,input)
    if input==1:
       a2=sigmoid(z2)
    else:
        a2=z2
    #comparing with target and calculating error
    totalError=abs(y-a2)
    avg=errorFunction(y,a2)
    #approximating 
    avgapprox = np.round(avg,3)
    print "Total Error in when number of nodes in hidden layer  is %d : \n" % HiddenLayerSize
    print avg
print "Optimal number of nodes in hidden layer is :  %d " % HiddenLayerSize

# when output node is linear it ovefit the result then the error is many less than when output node is sigmoid  
  




