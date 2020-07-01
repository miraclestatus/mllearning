import numpy as np
class Neural_Network():
    def __init__(self):
        self.theta=None
        self.b=None
        self.Cost=None
    def fit(self,X_train,y_train,p,alpha):
        for i in range(0,p):
            m=X_train.shape[0]
            self.theta=np.random.randn(1,X_train.shape[1])
            self.b=0
            Z=X_train.dot(self.theta.T)+self.b
            def sigmond(x):
                return 1/(1+np.exp(-x))
            A=sigmond(Z)
            self.Cost=y_train.dot(np.log(A))+(1-y_train).dot(np.log(1-A))/m
            dZ=A-y_train.T
            db=np.sum(dZ)/m
            dtheta=(X_train.T).dot(dZ)/m
            self.theta-=(dtheta*alpha).T
            self.b-=db*alpha
        return self
    def predict(self,X_test):
        Ap=X_test.dot(self.theta.T)+self.b
        for i in range(X_test.shape[1]):
            if Ap[0,i]>0.5:
                Ap[0,i]=1
            else:
                Ap[0,i]=0
        return Ap
    def score(self,X_test,y_test):
        return np.sum(self.predict(X_test)-y_test)
