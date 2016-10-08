import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def f_x(x,a,b):
    y = a+b*x
    return y
	
v = [890,3810,4630,4820,5230,7500,11800,19600,2350,630]
m = [12.5,15.5,15.4,16.0,16.4,17.0,18.0,19.0,13.8,11.6]
num_v = [7,5,4,2,4,3,1,1,16,21]
M=-13.8
N = len(v)
#2 free parameters
m_val=2

#...........................part a............................................
v_log = []
for i in v:
    v_log.append(math.log(i))
sum_m=0.0
sum_v =0.0	
sum_n=0.0
for i,j,k in zip(m,v_log,num_v):
    sum_m+=i
    sum_v+=j	

mean_m=sum_m/N
mean_v=sum_v/(N)

cov=0.0
var=0.0

for i,j in zip(m,v_log):
    cov+=(i-mean_m)*(j-mean_v)
    var+=(i-mean_m)**2
cov=cov/(N-1)
var = var/(N-1)
	
b = cov/var

a=mean_v-b*mean_m

print "b"
print b	
print "a"
print a

f_m=[]
for i in m:
    f_m.append(f_x(i,a,b))
	
#plotting
plt.plot(m,f_m,color='purple')
plt.scatter(m,v_log,color='green')
plt.legend(['best-fit\na={0}\nb={1}'.format(a,b),'data'],loc='upper left')
plt.xlabel('m')
plt.ylabel('log(v)')
plt.title('8.1 pt. (a)')
plt.savefig('8_1_plot.png')
#plt.show()
    
#........................part b............................................

m_s_var=0.0

for i,j in zip(f_m,v_log):
    m_s_var+=(j-i)**2
m_s_var=m_s_var/(N-m_val)

print 'model sample variance'
print m_s_var
print 'model error'
model_error=math.sqrt(m_s_var)
print model_error

delta=0.0
sum_sqrd=0.0
for i in m:
    sum_sqrd+=i
sum_sqrd=sum_sqrd**2
m_sqrd=0.0
for i in m:
    m_sqrd+=i**2
delta=(1/(model_error**4))*(N*m_sqrd-sum_sqrd) 

var_a=(1/(delta*m_s_var))*m_sqrd

var_b=N/(delta*m_s_var)
m_sum=0.0
for i in m:
    m_sum+=i
var_ab = -(1/(delta*m_s_var))*m_sum

correl = var_ab/(math.sqrt(var_a)*math.sqrt(var_b))

print 'a variance'
print var_a
print 'a error'
print math.sqrt(var_a)
print 'b variance'
print var_b
print 'b error'
print math.sqrt(var_b)
print 'covariance'
print var_ab
print 'correlation coefficient (a, b)'
print correl

#.........................part c...........................................
chi_sqrd=0.0
for i,j in zip(v_log,m):
    chi_sqrd+=(i-a-b*j)**2/m_s_var
	
print 'miminum chi squared using a and b'
print chi_sqrd
  	
	
