
# coding: utf-8

# In[10]:


# Product Function gradient differential components
def mul (a,b):
    return a * b

a=1
b=1
da = 1
db = 1

diffa = b * da
dfxdb = a * db


# In[13]:


# Add function gradient differential components
def add (a,b):
    return a +b

a=1
b=1
da = 1
db = 1

dfda = 1
dfdb = 1


# In[17]:


# Two additions gradient differential components
a=1
b=1
c=1

# the Two additions
q=a+b
x=q+c

# Backward Pass
# Last Add
dfdq = 1
dfdc = 1

#First Add
dqda = 1
dqdb = 1
dfdq = 1
dfda = dqda * dfdq
dfdb = dqdb * dfdq

# because differential dfdq = 1 (the differential of a sum, whatever the variable is), backward gate local gradient
# will be entirely defined by gradient of local gate function


# In[20]:


# 1 addition and 1 multiplication
a=1
b=1
c=1

q = a * b
x = q + c

# Backward pass
# Last Add
dfdq = 1
dfdc = 1

# First Mult
dqda = b
dqdb = a
dfda = dqda * dfdq # dfda = b * dfdq
dfdb = dqdb * dfdq # dfdb = a * dfdq

# because differential dqda = b & dqdb = a, the gradient of a multiply gate will apply the opposite input as local gradient
# differential component


# In[24]:


# Neuron in 2 steps with the ax + by + c function and the sigmoid
# Sigmoid pre-requesite 
import math
def sigmoid(X):
    return 1 / (1 + math.exp(-X))

a=1
b=1
c=1
x=1
y=1

q = a * x + b * y + c
sig = sigmoid(q)

# Backward Pass
# Last Sigmoid gate
dsigdq = sig * (1 - sig)

# First Sigma gate
dfda = x * dsigdq
dfdb = y * dsigdq
dfdx = a * dsigdq
dfdy = b * dsigdq
dfdc = 1 * dsigdq


# In[ ]:


# a*a + b*b + c*c
a=1
b=1
c=1

# Backward pass
# 1 gate in all only
dfda = 2 * a
dfdb = 2 * b
dfdc = 2 * c


# In[26]:


# ((a * b + c) * d)^2

a=1
b=1
c=1
d=1

