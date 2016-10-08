import sys
import math
from numpy import matrix
from numpy import linalg
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

iris_setosa = np.genfromtxt('Iris_setosa.csv',delimiter=',')

sepal_length = iris_setosa[:,0]
sepal_width = iris_setosa[:,1]
petal_length = iris_setosa[:,2]
petal_width = iris_setosa[:,3]

y_x_data = np.column_stack((sepal_length,sepal_width,petal_length,petal_width))

N = len(petal_width)

beta = np.array([0,0,0,0],dtype=float)
a = np.array([0,0,0,0],dtype=float)
m=3
A = np.zeros((m+1,m+1), dtype=float, order='C')
indexer = np.arange(m+1)
data_indexer = np.arange(50)

beta[0] = np.sum(sepal_length)

#beta (1 x m)
for i in indexer:
    print i
    if i == 0:
	    continue
    sum=0.0
    x_col = y_x_data[:,i]
    for j in data_indexer:
	    sum+=(y_x_data[j][i]*y_x_data[j][0])
    beta[i]=sum
	
print 'beta'	
print beta

#A_(m+1 X m+1)
for i in indexer:
    for j in indexer:
		for k in data_indexer:
		    x_i = y_x_data[k][i]
		    x_j = y_x_data[k][j]		
		    if (i == 0):
			x_i = 1.0
		    if (j == 0):
                        x_j = 1.0			
                    A[i][j]+= (x_i*x_j)	
    
print 'A'
print A        

inv_A = linalg.inv(A)
print 'A_inv'
print inv_A	

print 'a'
a = np.matmul(beta,inv_A)
print a

uncert = []
for i in indexer:
    var  = inv_A[i][i]
    print var
    uncert.append(math.sqrt(var))

print 'uncertainties in best fit parameters'
print uncert


#.......................................part b.......................................................

beta = np.array([0,0],dtype=float)
a = np.array([0,0],dtype=float)
m=1
A = np.zeros((m+1,m+1), dtype=float, order='C')
indexer = np.arange(m+1)
data_indexer = np.arange(50)

beta[0] = np.sum(sepal_length)

#beta (1 x m)
for i in indexer:
    print i
    if i == 0:
	    continue
    sum=0.0
    x_col = y_x_data[:,i]
    for j in data_indexer:
	    sum+=(y_x_data[j][i]*y_x_data[j][0])
    beta[i]=sum
	
print 'beta'	
print beta

#A_(m+1 X m+1)
for i in indexer:
    for j in indexer:
		for k in data_indexer:
		    x_i = y_x_data[k][i]
		    x_j = y_x_data[k][j]		
		    if (i == 0):
			x_i = 1.0
		    if (j == 0):
                        x_j = 1.0			
                    A[i][j]+= (x_i*x_j)	
    
print 'A'
print A        

inv_A = linalg.inv(A)
print 'A_inv'
print inv_A	

print 'a'
a = np.matmul(beta,inv_A)
print a

uncert = []
for i in indexer:
    var  = inv_A[i][i]
    print var
    uncert.append(math.sqrt(var))

print 'uncertainties in best fit parameters'
print uncert


	 

    


	
    

