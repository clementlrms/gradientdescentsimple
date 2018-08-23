
# coding: utf-8

# In[23]:


# Pre-requesite testing with Sigmoid function of a Single Variable
import math

def sigmoid (X):
    return 1 / (1+math.exp(-X))

def sigmoidDerivative (X):
    return sigmoid(X) * (1 - sigmoid(X))


# In[24]:


# Unit class definition
class Unit():
    def __init__(self,val,xi_der):
        self.value = val
        self.xi_derivative = xi_der


# In[82]:


class ProductGate():

    # init the Add Gate with 2 input Units
    def __init__(self,U0,U1):
        self.u0 = U0
        self.u1 = U1

    # this function will compute the output value of the Product Gate which contain        
    def forward(self):
        self.outputUnit = Unit(self.u0.value * self.u1.value,0.0)
        return self.outputUnit

    # update the xi_derivative of the input Units - notice the value of xi_derivative 
    def backward(self):
        self.u0.xi_derivative = self.u0.xi_derivative + self.u1.value * self.outputUnit.xi_derivative
        self.u1.xi_derivative = self.u1.xi_derivative + self.u0.value * self.outputUnit.xi_derivative


# In[83]:


class AddGate():
    
    # init the Add Gate with 2 input Units
    def __init__(self,U0,U1):
        self.u0 = U0
        self.u1 = U1
    
    # compute the value of the output Unit
    def forward(self):
        self.outputUnit = Unit(self.u0.value + self.u1.value,0.0)
        return self.outputUnit
    
    # update the xi_derivative of the input Units - notice the value of xi_derivative = 1
    def backward(self):
        self.u0.xi_derivative = self.u0.xi_derivative + 1 * self.outputUnit.xi_derivative
        self.u1.xi_derivative = self.u1.xi_derivative + 1 * self.outputUnit.xi_derivative


# In[84]:


class SigmoidGate():
    
    # init the Sigmoid Gate with a single input Unit
    def __init__(self,U0):
        self.u0 = U0
        
    # compute the value of the output Unit
    def forward(self):
        self.outputUnit = Unit(sigmoid(self.u0.value),0.0)
        return self.outputUnit
    
    # update the xi_derivative of the input Unit
    def backward(self):
        self.u0.xi_derivative = self.u0.xi_derivative + sigmoid(self.u0.value) * (1-sigmoid(self.u0.value))


# In[93]:


# Implementation of all the units
w1 = Unit(1,0)
x1 = Unit(-1,0)
w2 = Unit(2,0)
x2 = Unit(3,0)
w3 = Unit(-3,0)

# Implementation of the Gates + Pass Forward

mul0 = ProductGate(w1,x1)
mul0.forward()
mul1 = ProductGate(w2,x2)
mul1.forward()
add0 = AddGate(mul0.outputUnit,mul1.outputUnit)
add0.forward()
add1 = AddGate(add0.outputUnit,w3)
add1.forward()
sig0 = SigmoidGate(add1.outputUnit)
sig0.forward()

#### at this stage, the forward pass has been made through all the gates and all outputs are known ####


# In[94]:


print("### output value result of each gate, mul0, mul1, add0, add1, sig0")
print(mul0.outputUnit.value)
print(mul1.outputUnit.value)
print(add0.outputUnit.value)
print(add1.outputUnit.value)
print(sig0.outputUnit.value)


# In[95]:


#### Implementing the backforward propagation of the Gradient ####


# In[96]:


# Checking current xi_derivative value for each Gate before Backward Pass
print("### xi_derivative value of the different gate before 1 pass backward")
print(sig0.outputUnit.xi_derivative)
print(add1.outputUnit.xi_derivative)
print(add0.outputUnit.xi_derivative)
print(mul1.outputUnit.xi_derivative)
print(mul0.outputUnit.xi_derivative)


# In[97]:


# Checking current xi_derivative value of initinal input before Backward Pass
print("### xi_derivative value of initial inputs before 1 pass backward")
print(w1.xi_derivative)
print(x1.xi_derivative)
print(w2.xi_derivative)
print(x2.xi_derivative)
print(w3.xi_derivative)


# In[98]:


# Update last unit (sigmoid) xi_derivative to start off the backpropagation
sig0.outputUnit.xi_derivative = 1
#Start off backpropagation in reverser oder with Sigmoid
sig0.backward()
# Update xi_derivative values of input units
add1.backward()
# Update xi_derivative values of input units
add0.backward()
# Update xi_derivative values of input units
mul1.backward()
# Update xi_derivative values of input units
mul0.backward()


# In[99]:


test = sigmoid(2)*(1-sigmoid(2))
print(test)


# In[101]:


# Check value of the xi_derivative of the differnt gate
print("### xi_derivative value of the different gate after 1 pass backward ###")
print("xi_derivative out of sig0:",sig0.outputUnit.xi_derivative)
print("xi_derivative out of add1:",add1.outputUnit.xi_derivative)
print("xi_derivative out of add0 & w3:", add0.outputUnit.xi_derivative," & ", w3.xi_derivative)
print("xi_derivative out of mul0 & mul1:", mul0.outputUnit.xi_derivative," & ", mul1.outputUnit.xi_derivative)
print("xi_derivative out of w2 & x2:", w2.xi_derivative," & ", x2.xi_derivative)
print("xi_derivative out of w1 & x1:", w1.xi_derivative," & ", x1.xi_derivative)


# In[102]:


# Check value of the xi_derivative of the different Unit
print("### xi_derivative value of initial inputs after 1 pass backward")
print(w1.xi_derivative)
print(x1.xi_derivative)
print(w2.xi_derivative)
print(x2.xi_derivative)
print(w3.xi_derivative)


# In[103]:


#### Implementing the Gradient Descent by updating values of the units ####
step_size = 0.01
x1.value = x1.value + x1.xi_derivative * step_size
w1.value = w1.value + w1.xi_derivative * step_size
x2.value = x2.value + x2.xi_derivative * step_size
w2.value = w2.value + w2.xi_derivative * step_size
w3.value = w3.value + w3.xi_derivative * step_size
print("### updated value with Gradient Descent ###")
print(w1.value)
print(w2.value)
print(w3.value)
print(x1.value)
print(x2.value)

