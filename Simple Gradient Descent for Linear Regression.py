
######## This script implements the Gradient Descent for a simple Linear Regression #########
# This script is based on data & methodolgy described here: https://machinelearningmastery.com/linear-regression-tutorial-using-gradient-descent-for-machine-learning/
# we want to define y = f(x) so that y = f(x) = w0 + w1*x
# y is therefore a linear combination of x, it is a single variable function of x


x=[1,2,3,4,5]
y=[1,3,3,2,5]


for xi in x:
    print(xi)

for yi in y:
    print(yi)


import matplotlib.pyplot as plt
plt.scatter(x,y)


#Stochastic Gradient Descent - the SDG will allow to determine best w0,w1 which have been previously introduced


# Pre-requesite
# Defining y as a linear function of x with w0 and w1 parameters
def estimateY (W0,W1,X):
    calculus = W0+W1*X
    return calculus

# Defining the error between true y value and the one estimated
def error(Y,ESTIMATEY):
    calculus = ESTIMATEY-Y
    return calculus

# Defining Gradient Descent Learning Rate
alpha=0.01

# Defining w0 update function
def w0update(w0old,err):
    calculus = w0old - alpha * err
    return calculus
    
# Defining w1 update function
def w1update(w1old,err,X):
    calculus = w1old - alpha * err * X
    return calculus


# Gradient Descent iteration 1

# Initiliazing w0 & w1 to 0
w0=0.0
w1=0.0

# Calculating the estimated Y
y0estim=estimateY(w0,w1,x[0])
y0estim

# Calculating the error between estimated and
error0=error(y[0],y0estim)
error0


# Calculating the Gradient Descent for w0 & w1
w0=w0update(w0,error0)
print(w0)

w1=w1update(w1,error0,x[0])
print(w1)


# Gradient Descent iteration 2

# Calculating the estimated Y
y1estim=estimateY(w0,w1,x[1])
y1estim


# Calculating the error between estimated and
error1=error(y[1],y1estim)
error1

# Calculating the Gradient Descent for w0 & w1
w0=w0update(w0,error1)
print(w0)

w1=w1update(w1,error1,x[1])
print(w1)


# Gradient Descent iteration 3
w0=0.0397
w1=0.0694
# Calculating the estimated Y
y2estim=estimateY(w0,w1,x[2])
y2estim

# Calculating the error between estimated and
error2=error(y[2],y2estim)
error2

# Calculating the Gradient Descent for w0 & w1
w0=w0update(w0,error2)
print(w0)

w1=w1update(w1,error2,x[2])
print(w1)


# Gradient Descent iteration 4

# Calculating the estimated Y
y3estim=estimateY(w0,w1,x[3])
y3estim

# Calculating the error between estimated and
error3=error(y[3],y3estim)
error3

# Calculating the Gradient Descent for w0 & w1
w0=w0update(w0,error3)
print(w0)

w1=w1update(w1,error3,x[3])
print(w1)



# Introducing Gradient Descent Iteration function that will update W0 and W1 automatically
def gradientDescent(w0old,w1old,X,Y):
    yestim=estimateY(w0old,w1old,X)
    errorMade=error(Y,yestim)
    w0=w0update(w0old,errorMade)
    w1=w1update(w1old,errorMade,X)
    w=[w0,w1]
    return w


# Doing 1 complete EPOCH = 1 complete pass through the whole Dataset = 5 Gradient Descent Iterations 

# Initiliazing w0 & w1 to 0
w0=0.0
w1=0.0

# Gradient Descent iteration 1
wNew=gradientDescent(w0,w1,x[0],y[0])
wNew

# Gradient Descent iteration 2
wNew=gradientDescent(wNew[0],wNew[1],x[1],y[1])
wNew

# Gradient Descent iteration 3
wNew=gradientDescent(wNew[0],wNew[1],x[2],y[2])
wNew

# Gradient Descent iteration 4
wNew=gradientDescent(wNew[0],wNew[1],x[3],y[3])
wNew

# Gradient Descent iteration 5
wNew=gradientDescent(wNew[0],wNew[1],x[4],y[4])
wNew



# Doing 1 complete EPOCH = 1 complete pass through the whole Dataset with a Gradient Descent loop

# Initiliazing w0 & w1 to 0
w0=0.0
w1=0.0
wNew=[w0,w1]

for i in range(0,5):
    wNew = gradientDescent(wNew[0],wNew[1],x[i],y[i])

print(wNew)


# Doing 4 EPOCHs = 4 complete pass through the whole Dataset with a Gradient Descent loop = 20 Gradient Descent Iteration

# Initiliazing w0 & w1 to 0
w0=0.0
w1=0.0
wNew=[w0,w1]
errorList=[]

# j index to implement 4 EPOCHs
for j in range(0,4):
    # i index to implement 5 Gradient Descent iteration standing for the Dataset size
    for i in range(0,5):
        wNew = gradientDescent(wNew[0],wNew[1],x[i],y[i])
        currentError=error(y[i],estimateY(wNew[0],wNew[1],x[i]))
        errorList.append(currentError)

print(wNew)
print(errorList)


# Printing error evolution over the Gradient Descent iteration
iterationList=list(range(1,21))
plt.plot(iterationList,errorList)


# Making Prediction with the final ML Model, which is the Linear Regression defined with wNew after 4 EPOCHs=20 iteration

# Printing best values for w0 and w1
print("Best values for w0 and w1 after 4 EPOCHs = 20 iterations")
print(wNew)
print("\n")

# Defining the ML Model which is the linear regression with previous best w0 & w1
def finalModel(X):
    calculus = wNew[0] + wNew[1]*X
    return calculus

# Making prediction using the ML Linear Regression Model
print("Predicted y Values")
for xi in x:
    print(finalModel(xi))
print("\n")
    
print("Actual y Values")
for yi in y:
    print(yi)
    

