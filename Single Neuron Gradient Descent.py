
# coding: utf-8

# In[8]:


# Pre-requesite testing with Sigmoid function of a Single Variable
import math

def sigmoid (X):
    return 1 / (1+math.exp(-X))

def sigmoidDerivative (X):
    return sigmoid(X) * (1 - sigmoid(X))


# In[9]:


# Defining a Unit by two components: output and gradient - a Unit is a generic container for any type of gate
# a Unit can then be instanciated with Sigmoid to do a Neuron or whatever other Gate Function

# Unit class definition
class Unit():
    def __init__(self,val,xi_der):
        self.value = val
        self.xi_derivative = xi_der
        

# Now each gate is represented by: 
# - an output Unit defined by a value and a derivative
# - a forward method to compute the value of the output unit
# - a backward method to update the xi_derivative of the input units
# Remark: the gradient of a Gate function is given by the different xi_derivative of input Units

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
        self.u0.xi_derivative = self.u0.xi_derivative + s                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  elf.u1.value * self.outputUnit.xi_derivative
        self.u1.xi_xi_derivative = self.u1.xi_derivative + self.u0.value * self.outputUnit.xi_derivative
        
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
        


# In[12]:


# Implementation of all the units
w1 = Unit(1,0)
x1 = Unit(-1,0)
w2 = Unit(2,0)
x2 = Unit(3,0)
w3 = Unit(-3,0)

# Implementation of the Gates

def neuronForward():
    w1x1 = ProductGate(w1,x1)
    w1x1.forward()

    w2x2 = ProductGate(w2,x2)
    w2x2.forward()

    w1x1w2x2 = AddGate(w1x1.outputUnit,w2x2.outputUnit)
    w1x1w2x2.forward()

    w1x1w2x2w3 = AddGate(w1x1w2x2.outputUnit,w3)
    w1x1w2x2w3.forward()

    sig = SigmoidGate(w1x1w2x2w3.outputUnit)
    sig.forward()
    print("### output value result for Sigmoid after 1 pass forward")
    print(sig.outputUnit.value)

neuronForward()

#### at this stage, the forward pass has been made through all the gates and all outputs are known ####


# In[15]:


# Implementation of all the units
w1 = Unit(1,0)
x1 = Unit(-1,0)
w2 = Unit(2,0)
x2 = Unit(3,0)
w3 = Unit(-3,0)

# Implementation of the Gates

w1x1 = ProductGate(w1,x1)
w1x1.forward()
w2x2 = ProductGate(w2,x2)
w2x2.forward()
w1x1w2x2 = AddGate(w1x1.outputUnit,w2x2.outputUnit)
w1x1w2x2.forward()
w1x1w2x2w3 = AddGate(w1x1w2x2.outputUnit,w3)
w1x1w2x2w3.forward()
sig = SigmoidGate(w1x1w2x2w3.outputUnit)
sig.forward()
print("### output value result for Sigmoid after 1 pass forward")
print(sig.outputUnit.value)


#### at this stage, the forward pass has been made through all the gates and all outputs are known ####


# In[16]:


print("value after 1 pass forward")
print(w1x1.outputUnit.value)
print(w2x2.outputUnit.value)
print(w1x1w2x2.outputUnit.value)
print(w1x1w2x2w3.outputUnit.value)
print(sig.outputUnit.value)


# In[17]:


#### Implementing the backforward propagation of the Gradient ####
# Implementing the backward propagation in reverser order from
# Notice that so far all the value have not been yet updated with the gradient
print("### xi_derivative value of the different gate before 1 pass backward")
print(sig.outputUnit.xi_derivative)
print(w1x1w2x2w3.outputUnit.xi_derivative)
print(w1x1w2x2.outputUnit.xi_derivative)
print(w2x2.outputUnit.xi_derivative)
print(w1x1.outputUnit.xi_derivative)


# In[ ]:


# Update last unit (sigmoid) xi_derivative to start off the backpropagation
sig.outputUnit.xi_derivative = 1
#Start off backpropagation in reverser oder with Sigmoid
sig.backward()
# Update xi_derivative values of input units
w1x1w2x2w3.backward()
# Update xi_derivative values of input units
w1x1w2x2.backward()
# Update xi_derivative values of input units
w2x2.backward()
# Update xi_derivative values of input units
w1x1.backward()


# In[19]:


# Check value of the xi_derivative of the differnt gate
print("### xi_derivative value of the different gate after 1 pass backward")
print(sig.outputUnit.xi_derivative)
print(w1x1w2x2w3.outputUnit.xi_derivative)
print(w1x1w2x2.outputUnit.xi_derivative)
print(w2x2.outputUnit.xi_derivative)
print(w1x1.outputUnit.xi_derivative)


# In[21]:


# Check value of the xi_derivative of the different Unit
print("### xi_derivative value of the different gate after 1 pass backward")
print(w1.xi_derivative)
print(x1.xi_derivative)
print(w2.xi_derivative)
print(x2.xi_derivative)
print(w3.xi_derivative)


# In[18]:




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

