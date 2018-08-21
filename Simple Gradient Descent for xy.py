
# coding: utf-8

# In[39]:


# Defining the f(x1,x2) = x1*x2 function
def x1x2Product (X1,X2):
    return X1*X2


# In[40]:


# Testing xyProduct
x1x2Product(2,4)

# the goal of the following strategies is to improve x1x2 product result by slighlty moving x1 and x2


# In[49]:


# Random Local Search - using random number generation for k = 100 iterations to identify best product than 2*4=8
# With this approach we'll see that moving randomly x1 and x2 using alpha rate = 0.01 will give a better result than 8

# Parameters initialization
alpha=0.01
x1=2
x2=4
bestx1=x1
bestx2=x2
bestx1x2Prod=-10000

# Building the loop to iterate 100 times
import random as rd
for k in range(0,100):
    x1_try = x1 + alpha * rd.uniform(-1,1)
    x2_try = x2 + alpha * rd.uniform(-1,1)
    x1x2Prod = x1x2Product(x1_try,x2_try)
    if(x1x2Prod > bestx1x2Prod):
        bestx1x2Prod = x1x2Prod
        bestx1=x1_try
        bestx2=x2_try
        print(k)
        print(bestx1x2Prod)
        print(bestx1,bestx2)


# In[59]:


# Numerical Gradient

# Parameters initialization
h=0.0001
x1=-2
x2=3
x1x2Prod = x1x2Product(x1,x2)

# Compute differential of f with respect to x1
x1ph = x1 + h;
x1phx2Prod = x1x2Product(x1ph,x2)
print("x1x2 product with x1 small variation")
print(x1phx2Prod)
x1_derivative = (x1phx2Prod - x1x2Prod)/h
print(x1_derivative) # to be noticed that this is very close to x2 value

# Compute differential of f with respect to x2
x2ph = x2 + h;
x1x2phProd = x1x2Product(x1,x2ph)
print("x1x2 product with x2 small variation")
print(x1x2phProd)
x2_derivative = (x1x2phProd - x1x2Prod)/h
print(x2_derivative) # to be noticed that this is very close to x1 value

print("\n")

# Finally computing the value of x1x2Product taking into account the Gradient
step_size = 0.01
x1 = x1 + step_size*x1_derivative
x2 = x2 + step_size*x2_derivative
x1x2New = x1x2Product(x1,x2)
print(x1x2New)

# Note that at this step we are just taking into account the Gradient Value instead of a random value
# We are not yet following the Descent of the Gradient which might consist on following th Gradient when it decreases
# This needs to be clarfied based on complementary comment

