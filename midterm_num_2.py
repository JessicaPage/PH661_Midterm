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

'''
y_best_fit = []
for i in data_indexer:

	y_best_fit.append(a[0] + a[1]*y_x_data[i][1] + a[2]*y_x_data[i][2] + a[3]*y_x_data[i][3])	
'''

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


'''
m_s_var=0.0
m_val=2

for i,j in zip(sepal_length,sepal_width):
    m_s_var+=(j-i)**2
m_s_var=m_s_var/(N-m_val)

print 'model sample variance'
print m_s_var
print 'model error'
model_error=math.sqrt(m_s_var)
print model_error

sum_y =0.0	
sum_x=0.0
for i,j in zip(sepal_length,sepal_width):
    sum_y+=i
    sum_x+=j	

mean_y=sum_y/N
mean_x=sum_x/(N)

cov=0.0
var=0.0

for i,j in zip(sepal_width,sepal_length):
    cov+=(i-mean_x)*(j-mean_y)
    var+=(i-mean_x)**2
cov=cov/(N-1)
var = var/(N-1)
	
b = cov/var

a=mean_y-b*mean_x

print "b"
print b	
print "a"
print a

def f_x(x,a,b):
    y = a+b*x
    return y

f_m=[]
for i in sepal_width:
    f_m.append(f_x(i,a,b))
	
#plotting
plt.plot(sepal_width,f_m,color='purple')
plt.scatter(sepal_width,sepal_length,color='green')
plt.legend(['best-fit\na={0}\nb={1}'.format(a,b),'data'],loc='upper left')
plt.xlabel('sepal width')
plt.ylabel('sepal_length')
plt.title('9.1 pt. (b)')
plt.savefig('9_1_plot.png')
plt.show()	


# trying original method ignoring 

'''
	 

    


	
    

