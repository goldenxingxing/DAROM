import numpy as np
from numpy import linalg as LA

import math
class DLA:
    def __init__(self, epsilon, _lambda=0, gamma=0.1):

        self._lambda = _lambda
        self._epsilon = epsilon
        self.gamma = gamma


    def check_convergence(self, x,y,epsilon=1e-10):
        return LA.norm(x-y)<epsilon
    def dynamic(self, U, b, C):

        n, m = U.shape

        X = np.zeros((n, m))

        unit_num = int(self._epsilon * m)

        for j in range(unit_num, m):

            #if j == unit_num:
            if j % unit_num == 0:
                trainU = U[:, j - unit_num:j]
                trainb = self._epsilon * b*(1-self._epsilon)
                trainC = C[:, j - unit_num:j]
                # print(trainb)
                alpha = self.fit_alpha(trainU, trainb, trainC)
            p = U[:,j]-C[:,j]*alpha
            desc = np.argsort(-p)
            for i in desc:
                if U[i,j]-C[i,j]*alpha[i] and np.sum(C[i,:]*X[i,:])+C[i,j]<=b[i]:
                    X[i,j] = 1
                    break
#            i = np.argmax(U[:,j]-C[:,j]*alpha)
#            if U[i,j]-C[i,j]*alpha[i]>0 and np.sum(C[i,:]*X[i,:])+C[i,j]<=b[i]:
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
                a[i] = max(0, a[i])
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
        #return np.sum(a > 0), np.sum([j for j in a if j >0])  
        
        


class DLA_paper:
    def __init__(self, epsilon, _lambda=0, gamma=0.1):

        self._lambda = _lambda
        self._epsilon = epsilon
        self.gamma = gamma


    def check_convergence(self, x,y,epsilon=1e-10):
        return LA.norm(x-y)<epsilon
    def dynamic(self, U, b, C):

        n, m = U.shape

        X = np.zeros((n, m))

        unit_num = int(self._epsilon * m)
        r = 0
        for j in range(unit_num, m):

            if j == 2**r*unit_num:
                r += 1
                trainU = U[:, :j]
                hl = self._epsilon*math.sqrt(float(m)/j)
                trainb = (1-hl)*j/m* b
                trainC = C[:, :j]
                # print(trainb)
                alpha = self.fit_alpha(trainU, trainb, trainC)
            p = U[:,j]-C[:,j]*alpha
            desc = np.argsort(-p)
            for i in desc:
                if U[i,j]-C[i,j]*alpha[i]>0 and np.sum(C[i,:]*X[i,:])+C[i,j]<=b[i]:
                    X[i,j] = 1
                    break

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
                a[i] = max(0, a[i])
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
        #return np.sum(a > 0), np.sum([j for j in a if j >0])  
        
        
