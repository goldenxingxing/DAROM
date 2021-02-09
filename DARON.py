import numpy as np
from numpy import linalg as LA

from RDLAModel import *
from DLA_Model import *

class ARRIVAL:
    def __init__(self,  degree, epsilon):

        self.degree = degree
        self._epsilon = epsilon
    def output(self, n,m):
        unit_num = int(self._epsilon * m)
        U = np.zeros((n,m))
#        choice = np.array(range(1,m/unit_num))
        for j in range(m):
            batch_j = j/unit_num + 1
            rnd = self.degree * batch_j
#            U[:,j] = np.random.beta(1+rnd,3) 
#            U[:,j] = np.random.gamma(1+rnd,3)
            U[:,j] = np.random.exponential(1+rnd)
        return U

class DARON:
    def __init__(self, _lambda, epsilon, gamma=0.1):

        self._lambda = _lambda
        self._epsilon = epsilon
        self.gamma = gamma


    def check_convergence(self, x,y,epsilon=1e-12):
        return LA.norm(x-y)<epsilon
    def dynamic(self, U, b, C):

        n, m = U.shape

        X = np.zeros((n, m))

        unit_num = int(self._epsilon * m)

        for j in range(unit_num, m):

            #if j == unit_num:
            if j % unit_num == 0:
                trainU = U[:, j - unit_num:j]
                trainb = self._epsilon * b
                trainC = C[:, j - unit_num:j]
                # print(trainb)
                alpha = self.fit_alpha(trainU, trainb, trainC)
            p = U[:,j]-C[:,j]*alpha
            desc = np.argsort(-p)
            for i in desc:
                if U[i,j]-C[i,j]*alpha[i]>0 and np.sum(C[i,:]*X[i,:])+C[i,j]<=b[i]:
                    X[i,j] = 1
                    break
#            i = np.argmax(U[:,j]-C[:,j]*alpha)
#            if U[i,j]-C[i,j]*alpha[i]>0:
#                X[i,j] = 1

        # print(np.sum(curr_b == 0) / curr_b.shape[0])
        return X
    def fit_alpha(self, U, b, C):
        n, m = U.shape
        a = np.ones(n)*0.1
        assert n==len(b)
        for i in range(n):
            while True:
                a_prev = a
                I_i = self.set_i(U,C,a,i)
                l2 = LA.norm(a)
                g_ai = b[i]-np.sum([C[i,j] for j in I_i])-self._lambda*(1/l2-sum(a)*a[i]/(l2**3))
                a[i] -= self.gamma*g_ai
                
                if self.check_convergence(a_prev,a):
                    break
        a[a<0]=0
        return a
    
    
    def calculate_total_reward(self,U, X):

        return np.sum((U * X))
                   
            
    def set_i(self, U, C,a,i):
        n, m = U.shape
        I_i = []
        for j in range(m):
            u = U[:,j]
            c = C[:,j]
            if np.argmax(u-c*a) == i:
                I_i.append(j)
        return I_i
    def show_violation(self, C, X, b):
        c = np.zeros(len(b))
        for i in range(len(b)):
            c[i] = np.sum(C[i,:]*X[i,:])
        a = c-b
        return np.sum(a > 0)
        #return np.sum(a > 0), np.sum([j for j in a if j >0])
            
         
        
class OT_PD:
    def __init__(self,epsilon, _lambda=0, gamma=0.1 ):

        self._lambda = _lambda
        self._epsilon = epsilon
        self.gamma = gamma


    def check_convergence(self, x,y,epsilon=1e-12):
        return LA.norm(x-y)<epsilon
    def dynamic(self, U, b, C):

        n, m = U.shape

        X = np.zeros((n, m))

        unit_num = int(self._epsilon * m)

        for j in range(unit_num, m):

            if j == unit_num:
                trainU = U[:, j - unit_num:j]
                trainb = self._epsilon * (1-self._epsilon)*b
                trainC = C[:, j - unit_num:j]
                # print(trainb)
                alpha = self.fit_alpha(trainU, trainb, trainC)
            p = U[:,j]-C[:,j]*alpha
            desc = np.argsort(-p)
            for i in desc:
                if U[i,j]-C[i,j]*alpha[i]>0 and np.sum(C[i,:]*X[i,:])+C[i,j]<=b[i]:
                    X[i,j] = 1
                    break
#            i = np.argmax(U[:,j]-C[:,j]*alpha)
#            if U[i,j]-C[i,j]*alpha[i]>0:
##                X[i,j] = 1
#            p = U[:,j]-C[:,j]*alpha
#            desc = np.argsort(-p)
#            for i in desc:
#                if U[i,j]-C[i,j]*alpha[i]>0 and np.sum(C[i,:]*X[i,:])+C[i,j]<=b[i]:
#                    X[i,j] = 1

        # print(np.sum(curr_b == 0) / curr_b.shape[0])
        return X
    def fit_alpha(self, U, b, C):
        n, m = U.shape
        a = np.ones(n)*0.1
        assert n==len(b)
        while True:
            a_prev = a
            for i in range(n):               
                I_i = self.set_i(U,C,a,i)
                g_ai = b[i]-np.sum([C[i,j] for j in I_i])
                a[i] -= self.gamma*g_ai
            a[a<0]=0   
            if self.check_convergence(a_prev,a):
                break     
        return a
    
    
    def calculate_total_reward(self,U, X):

        return np.sum((U * X))
                   
            
    def set_i(self, U, C,a,i):
        n, m = U.shape
        I_i = []
        for j in range(m):
            u = U[:,j]
            c = C[:,j]
            if np.argmax(u-c*a) == i:
                I_i.append(j)
        return I_i
    def show_violation(self, C, X, b):
        c = np.zeros(len(b))
        for i in range(len(b)):
            c[i] = np.sum(C[i,:]*X[i,:])
        a = c-b
        return np.sum(a > 0)       
        
        
e=0.1   
n=50
m=200
u=ARRIVAL(0.3,e)

U=u.output(n,m)
b=np.ones(n)
C=np.ones((n,m))*.4
#U=np.random.rand(n,m)
#C=U
#U2=U
from matplotlib import pyplot as plt 
 
#   
#
#plt.hist(U, bins=50) 
#
#plt.gca().set(title='Frequency Histogram', ylabel='Frequency');






dla_p=DLA_paper(e) 
dla_e=DLA(e) 




delta=[1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000]

#rdla=RDLA(delta,e)



lmbd=[1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,2000,3000,4000]
#lmbd=[2e6,3e6,4e6,5e6,1e7]

#daron=DARON(lmbd,e)

ot_pd=OT_PD(e)

Return_ot =[]
Violation_ot = []

Return_dla =[]
Violation_dla = []
Return_dla_p =[]
Violation_dla_p = []

Return_daron =[]
Violation_daron = []

Return_rdla =[]
Violation_rdla = []

times = 1
for i in range(times):
    x_o=ot_pd.dynamic(U,b,C)
    Return_ot.append(ot_pd.calculate_total_reward(U,x_o))    
    Violation_ot.append(ot_pd.show_violation(C,x_o,b)   ) 
    x_e=dla_e.dynamic(U,b,C)
    Return_dla.append(dla_e.calculate_total_reward(U,x_e))    
    Violation_dla.append(dla_e.show_violation(C,x_e,b)   )
    x_p=dla_p.dynamic(U,b,C)
    Return_dla_p.append(dla_p.calculate_total_reward(U,x_p))    
    Violation_dla_p.append(dla_p.show_violation(C,x_p,b)   )    
print('OT_PD: ', np.mean(Return_ot))
#print('DLA_e: ', np.mean(Return_dla))
print('DLA_p: ', np.mean(Return_dla_p))
for i in delta:
    print('RDLA')
    rdla=RDLA(i,e)
    x_r=rdla.dynamic(U,b,C)
    Return_rdla.append(rdla.calculate_total_reward(U,x_r))
    Violation_rdla.append(rdla.show_violation(C,x_r,b) )    
print('RDLA: ', Return_rdla)
        
#        
        
for i in lmbd:
    daron=DARON(i,e)
    x_d=daron.dynamic(U,b,C)
    Return_daron.append(daron.calculate_total_reward(U,x_d)    )    
    Violation_daron.append(daron.show_violation(C,x_d,b)   )
print('DARON:', Return_daron)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        