import numpy as np

from Plot.py import Plot

from __future__ import division, print_function

import cvxopt


# Hide cvxopt output
cvxopt.solvers.options['show_progress'] = False




"""The Linear kernel is the simplest kernel function. It is given by 
the inner product <x,y> plus an optional constant c.
Linear Kernel= k(x,y) = x^T * y + c """

def linear_kernel(**kwargs):
#kwargs allow you to pass a variable number of arguments to a function
    def f(x1, x2):
        return np.inner(x1, x2)
        #.inner(): gives Inner product of two arrays.
    return f



"""Polynomial kernel is a non-stationary kernel. Polynomial kernels are well suited 
for problems where all the training data is normalized."""
def polynomial_kernel(degree, constant, **kwargs):
    
     # k(x,y) = ( alpha * x^T * y + c)^d
     # alpha = slope, c = constant, d =degree
    def f(x1, x2):
        return (np.inner(x1, x2) + constant)**degree
    return f



#commonly used in support vector machine classification
def radius_basis_func_kernel(gamma, **kwargs):
    
    #RBF kernel = k(x1,x2) = exp^(- gamma * ||x1 - x2||^2), gamma > 0
    def f(x1,x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    return f




#Normalize data set X by dividing it with L2 norm
def normalize_rescale(X, axis = -1, order =2):
    l2_norm = np.atleast_1d(np.linalg.norm(X, order, axis))
  
    
    #L2 norm is square root of sum of squared elements of a vector 
    l2_norm[l2_norm == 0]=1
        
    #dividing data set with L2 norm of each of its vector to normalize
    return X / np.expand_dims(l2_norm, axis)



#computes accuracy of y_true training data w.r.t predicted y
def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis = 0)/ len(y_true)
    return accuracy
    #can use : " accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)" method of sklearn directly





#Splits data set into training and test subsets
def splitting_data(X, y, test_size = 0.5, shuffle = True, seed = None):
    
    if shuffle:
        X, y = shuffle(X, y , seed)
        
    #split training data from test data by test_size factor.
    split = len(y) - int( len(y) // (1 / test_size) )
   
    X_train_set = X[ :split] #(elements before split factor)
    X_test_set = X[split: ] #(elements after split factor)
    y_test_set = y[split: ]
    y_train_set = y[ :split]
    
    return X_train_set, X_test_set, y_train_set, y_test_set
    
    #can also use train_test_split() method of sklearn directly



#Support Vector Machine classifier
#Makes use of cvxopt to solve quadratic optimization problem
class SupportVectorMachine(object):
    """Required Parameters:
        C (float):
            Penalty term
        kernel function:
            Linear, polynomial or RGB depending on the data
            
        degree (int):
            The degree of the polynomial kernel. Will be ignored by the other
            kernel functions.
            
        gamma (float):
            Used in RGB kernel
            
        bias constant (float):
            Bias term used in polynomial kernel """
     
    
    def _init_(self, C=1, kernel = radius_basis_func_kernel, degree = 4, gamma = None, bias_constant = 4):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.bias_constant = bias_constant
        self.lagrange_multipliers = None
        self.support_vec = None
        self.support_vec_labels = None
        self.intercept = None
        
        
        
    def fit(self, X, y):
        samples, features = np.shape();
        
        #if gamma is None set it to the default value 1/n
        if not self.gamma:
            self.gamma = 1/features
          
            
        #Initialize the kernel with required parameters
        self.kernel = self.kernel(
                degree = self.degree,
                gamma = self.gamma)
               
        #create zero's matrix
        self.kernel = np.zeroes((samples, samples))
        
        
   
        
        
        
        
        
    
    


