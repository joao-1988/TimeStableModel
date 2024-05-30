import numpy as np

class Logistic():

  def __init__(self,intercept=True,max_iter=1000,tol=1e-6,learning_rate=0.1,random_state=None):
    self.intercept = intercept
    self.max_iter = max_iter
    self.tol = tol
    self.learning_rate = learning_rate
    self.random_state = random_state