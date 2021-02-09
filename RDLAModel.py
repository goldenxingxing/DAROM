import numpy as np

import matplotlib.pyplot as plt


from numpy import linalg as LA

class RDLA:

    def __init__(self, delta,_lambda=10, epsilon=0.1,  eta=0.1, maxIter=200):

        self._lambda = _lambda
        self._epsilon = epsilon
        self._delta = delta
        self._eta = eta

        self._maxIter = maxIter
        self._R = None
        self._b = None
        self._M = -1
        self._T = -1
        self._logT = -1
        self._alpha = None

        self._no = 0
        
    def check_convergence(self,x,y,epsilon=1e-12):
        return LA.norm(x-y)<epsilon

    def dynamic(self, R, b, C):
#        print('start')
        M, T = R.shape

        X = np.zeros((M, T))

        unit_num = int(self._epsilon * T)

        curr_b = np.copy(b)
        for t in range(unit_num, T):

            if t % unit_num == 0:
#                print(t)
                trainR = R[:, t - unit_num:t]
                trainb = self._epsilon * b
                trainC = C[:, t - unit_num:t]
                # print(trainb)
                alpha = self.fit(trainR, trainb,trainC)
                Z = (R - C*alpha.reshape((M, 1))) / self._lambda
                maxZ = np.max(Z, axis=0)
                Z = np.exp(Z - maxZ)
                sumZ = np.sum(Z, axis=0)
                Z = Z / sumZ
                # curr_b = 1 / (1 / self._epsilon - 1) * b

            # print/(curr_b)
            Zt = zip(range(M), Z[:, t].tolist(), R[:, t].tolist())
            Zt = sorted(Zt, key=lambda a: a[1], reverse=True)
            chosen_ad = -1
            for i in range(M):
                if Zt[i][2] <= 0:
                    continue
                if curr_b[Zt[i][0]] > 0:
                    chosen_ad = Zt[i][0]
                    curr_b[Zt[i][0]] -= C[chosen_ad][t]
                    break
            if chosen_ad != -1 and curr_b[chosen_ad]>=0:
                X[chosen_ad][t] = 1

        # print(np.sum(curr_b == 0) / curr_b.shape[0])
        return X

    def fit(self, R, b,C):

        self._R = np.copy(R)
        self._b = np.copy(b)
        self._C = np.copy(C)
        self._M, self._T = self._R.shape
        self._logT = np.log(self._T)

       # print(self._M, self._T)

        alpha = self._alternating_optimization()

        return alpha

    def _alternating_optimization(self):

        alpha = np.zeros(self._M)
        p = np.ones(self._T) / self._T
        loss = self._calculate_dual_loss(alpha, p)

        curr_i = 0
        while True:

            # X_it and at
            X = np.zeros((self._M, self._T))
            a = np.zeros(self._T)
            for t in range(self._T):

                Zt = (self._R[:, t] - self._C[:,t]*alpha) / self._lambda
                maxZt = np.max(Zt)
                Zt = np.exp(Zt - maxZt)
                sumZt = np.sum(Zt)
                X[:, t] = Zt / sumZt
                a[t] = maxZt + np.log(sumZt)
            # print(self._b / self._T / np.dot(X, p))

            # p_t and alpha_i
            gamma = self._lambda / (self._T * self._logT) * np.sum(a)
            inner_loss = self._calculate_inner_dual_loss(gamma, a)
            curr_j = 0
            while True:
                # print(gamma)
                p = - self._lambda / gamma * a
                # print(p)
                maxp = np.max(p)
                # print('sump:', maxp)
                p = np.exp(p - maxp)
                sump = np.sum(p)
                # print('sump:', sump)
                p = p / sump
                logsump = maxp + np.log(sump)
                ggamma = self._delta - self._logT + logsump + self._lambda / gamma * np.dot(p, a)
                gamma -= self._eta * ggamma
                gamma_rec = gamma
                gamma = max(0, gamma)
                if gamma==0:
                    gamma = gamma_rec
                    break

                prev_inner_loss = inner_loss
                inner_loss = self._calculate_inner_dual_loss(gamma, a)
                # print(inner_loss)
                # print(np.dot(p, np.log(self._T * p)))
                if curr_j == 50*self._maxIter or self.check_convergence(prev_inner_loss, inner_loss, epsilon=1e-5):
                    break
                else:
                    curr_j += 1
                # print(gamma)
            # print(gamma)
            p = - self._lambda / gamma * a
            # maxp = np.max(p)
            # print(np.min(p - maxp))
            p = np.exp(p)
            # print(np.sum(p == 0))
            sump = np.sum(p)
            p = p / sump
            # print(np.dot(p, np.log(self._T * p)))
            for i in range(self._M):
                wi = np.dot(X[i, :], p)
                alpha[i] = max(0, alpha[i] - self._lambda * np.log(self._b[i] / (self._T * wi)))

            # print(alpha)
            prev_loss = loss
            loss = self._calculate_dual_loss(alpha, p)
            # print('{}: {}'.format(curr_i, loss))
            if curr_i == self._maxIter or self.check_convergence(prev_loss, loss):
                break
            else:
                curr_i += 1
        self._p = p

#        fonts = {'family': 'Times New Roman',
#                 'weight': 'normal',
#                 'size': 17,
#                 }
#
#        plt.figure(21, figsize=(8, 8))
#        x = list(range(self._T))
#
#        sortp = sorted(zip(x, list(p)), key=lambda a: a[1])
#        ax1 = plt.subplot(211)
#        if self._no == 0:
#            ax1.set_ylabel('weight', fonts)
#        ax1.plot(x, [b * self._T for b in [a[1] for a in sortp]], 'r', linewidth=3.0, label='weight')
#        ax1.plot(x, [1] * self._T, 'b--', linewidth=3.0)
#        ax1.legend(loc='lower right')
#        labels = ax1.get_xticklabels() + ax1.get_yticklabels()
#        [label.set_fontname('Times New Roman') for label in labels]
#        plt.xticks(fontsize=15)
#        plt.yticks(fontsize=15)
#
#        top = 3
#        meanR_list_top = []
#        # meanR_list_all = []
#        # meanR_list_out = []
#        for iii, ppp in sortp:
#            # print(sorted(self._R[:, iii].tolist(), reverse=True)[:top])
#            meanR_list_top.append(sum(sorted(self._R[:, iii].tolist(), reverse=True)[:top]) / top)
#            # meanR_list_all.append(sum(self._R[:, iii].tolist()) / self._M)
#            # meanR_list_out.append(sum(self._R[:, iii].tolist()) / np.sum(self._R[:, iii] > 0))
#        ax2 = plt.subplot(212)
#        if self._no == 0:
#            ax2.set_ylabel('mean of top-3 bids', fonts)
#        ax2.plot(x, meanR_list_top, color='orange', linewidth=3.0, label='mean of top-3 bids')
#        ax2.legend(loc='upper left', bbox_to_anchor=(0.2, 1.0))
#        labels = ax2.get_xticklabels() + ax2.get_yticklabels()
#        [label.set_fontname('Times New Roman') for label in labels]
#        plt.xticks(fontsize=15)
#        plt.yticks(fontsize=15)
#        # plt.plot(list(range(len(meanR_list_all))), meanR_list_all, 'b')
#        # plt.plot(list(range(len(meanR_list_out))), meanR_list_out, 'g')
#
#        nonzero_list = []
#        for iii, ppp in sortp:
#            nonzero_list.append(np.sum(self._R[:, iii] > 0))
#        window = 200
#        nonzero_list = [max(nonzero_list[:100])] * (window - 1) + nonzero_list
#        nonzero_list = [sum(nonzero_list[jjj - window: jjj]) / window for jjj in range(window, len(nonzero_list) + 1)]
#        ax3 = ax2.twinx()
#        ax3.set_xlabel('Users', fonts)
#        if self._no == 2:
#            ax3.set_ylabel('candidating bidders number', fonts)
#        ax3.plot(list(range(len(nonzero_list))), nonzero_list, color='dodgerblue', linewidth=3.0, label='candidating bidders number')
#        ax3.legend(loc='upper right')
#        labels = ax3.get_xticklabels() + ax3.get_yticklabels()
#        [label.set_fontname('Times New Roman') for label in labels]
#        plt.xticks(fontsize=15)
#        plt.yticks(fontsize=15)
#
#        # plt.show()
#        plt.savefig('weights_{}.pdf'.format(self._no), format='pdf', bbox_inches='tight')
#        plt.close()
#        self._no += 1

        return alpha

    def _calculate_primal_loss(self, alpha, p):

        Z = (self._R - self._C*alpha.reshape((self._M, 1))) / self._lambda
        maxZ = np.max(Z, axis=0)
        Z = np.exp(Z - maxZ)
        sumZ = np.sum(Z, axis=0)
        X = Z / sumZ

        loss1 = np.sum(p * (self._R * X))
        loss2 = 0
        for t in range(self._T):
            entropy = - np.dot(X[:,t], np.log(X[:,t] + 1e-8))
            loss2 += p[t] * entropy
        loss2 *= self._lambda

        loss = loss1 + loss2

        return loss

    def _calculate_dual_loss(self, alpha, p):

        Z = (self._R - self._C*alpha.reshape((self._M, 1))) / self._lambda
        maxZ = np.max(Z, axis=0)
        Z = maxZ + np.log(np.sum(np.exp(Z - maxZ), axis=0))
        loss1 = self._lambda * np.dot(p, Z)
        loss2 = np.dot(alpha, self._b) / self._T

        loss = loss1 + loss2
        return loss

    def _calculate_inner_dual_loss(self, gamma, a):

        p = - self._lambda / gamma * a
        maxp = np.max(p)
        p = np.exp(p - maxp)
        sump = np.sum(p)
        logsump = maxp + np.log(sump)

        return gamma * (self._delta - self._logT + logsump)
    def show_violation(self, C, X, b):
        c = np.zeros(len(b))
        for i in range(len(b)):
            c[i] = np.sum(C[i,:]*X[i,:])
        a = c-b
        return np.sum(a > 0) 
    @staticmethod
    def draw_assignment(ax, X, b):

        assign_percentage = np.sum(X, axis=1) / b
        ax.bar(range(assign_percentage.shape[0]), assign_percentage)

    @staticmethod
    def calculate_total_reward(R, X):

        return np.sum((R * X))


