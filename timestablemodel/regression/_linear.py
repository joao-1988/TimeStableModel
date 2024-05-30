import numpy as np

class Linear():

  def __init__(self,intercept=True,max_iter=1000,tol=1e-6,learning_rate=0.1,random_state=None):
    self.intercept = intercept
    self.max_iter = max_iter
    self.tol = tol
    self.learning_rate = learning_rate
    self.random_state = random_state

  def predict(self,X):
    return self.intercept_ + X.dot(self.coef_)
  
  def __gradient_descent(self,X,y):
    for i in range(1,self.max_iter+1) :
      predictions = self.predict(X)
      err = predictions - y 
      grad_coef = np.dot(X.T, err)/self.n_
      self.coef_ -= grad_coef*self.learning_rate
      if self.intercept == True:
        grad_int = np.sum(err)/self.n_
        self.intercept_ -= grad_int*self.learning_rate

      c_tol = np.max(np.abs(grad_coef*self.learning_rate))/np.max(np.abs(self.coef_)) 
      if c_tol <= self.tol:
        break
    self.n_iter_ = i
    return

  def fit(self,X,y):
    
    np.random.seed(self.random_state)
    self.n_, self.d_ = X.shape
    self.coef_ = np.random.normal(0,1,self.d_)
    self.intercept_ = 0

    self.__gradient_descent(X,y)

    return self