import numpy as np

def ackley(x, a=20, b=0.2, c=2*np.pi):
    """
    x: vector of input values
    """
    d = len(x) # dimension of input vector x
    sum_sq_term = -a * np.exp(-b * sum(x*x) / d)
    cos_term = -np.exp(sum(np.cos(c*x) / d))
    return a + np.exp(1) + sum_sq_term + cos_term

# We want to optimize Ackley for two variables
print(ackley(np.array([1,2])))
print(ackley(np.array([0,0])))
