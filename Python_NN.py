# Some imports - - -
import matplotlib.pyplot as plt # to plot figures
import numpy as np # for scientific computing
import pandas as pd # for data visualisation
from scipy.optimize import minimize # for training


# Settings - - -
plt.rcParams.update({'figure.max_open_warning': 0})
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
pd.set_option('display.width', 1000)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Functions
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# Just a shorthand for printing matrices
def dprint(mat):
    print(pd.DataFrame(mat))

# computes the sigmoid value of the input array
def sigmoid(z):
    g = np.array(1/(1+np.exp(z)))
    return g

# initializes random weights between two layers having sizes L_in and L_out. It will automatically add a bias node to the L_in layer.
def randInitThetas(L_in, L_out):
    ep = np.sqrt(6)/(np.sqrt(L_in+L_out))
    t = np.random.rand(L_out, L_in + 1)*2*ep-ep
    return np.array(t)

# Computes the sum of the weights for the regularization term. Takes a vector of weights.
def thetaSum(t):
    t = t[:,1:]*t[:,1:]
    t = np.sum(t)
    return t

# Peforms forward pass calcuations for each layer in the Neural Network besides the input layer
# Inputs : "i" = the feature values for the layer, "t" = the theta weights connecting between this layer and the next, "addbias" = does the layer include a bias node? 1 for yes, 0 for no. Don't add a bias if you're running this function to determine the output of the neural network (ie: the final layer)
def layCalc(i, t, addbias):
    z = np.dot(i,t.T)
    a = sigmoid(z)

    if addbias == 1:
        m = len(a)
        bias = np.matrix(np.ones((m)))
        a = np.concatenate((bias.T, a), 1)

    a = np.array(a)
    z = np.array(z)
    return a, z

# The cost function that we want to minimize WRT the weights.
# Inputs: "nn_params" = the initial randomly initialized weights, "il" = input layer size, "ol" = output layer size, "X" = the training data, "y" = the ground truth labels for the training samples, "lam" = the regularization term, "h1, h2....hn" = the size of each hidden layer. Here we are only using one
def nnCostFunction(nn_params, il, ol, X, y, lam, h1):

    # region - Re-constructing Thetas from nn_params
    s1 = (il + 1) * h1
    s2 = (h1 + 1) * ol

    # placeholders for the weights
    a = np.array(nn_params[0:s1])
    b = np.array(nn_params[s1:s1 + s2])

    # The weights for each layer, with the bias node added
    Theta1 = np.reshape(a, (h1, il + 1))
    Theta2 = np.reshape(b, (ol, h1 + 1))
    # endregion

    # region - Layer Calcs
    m = len(X) # number of training samples
    bias = np.matrix(np.ones((m))) # appending another feature column for the bias node to correspond to
    X = np.concatenate((bias.T, X), 1)

    # input -> h1
    [a2, z2] = layCalc(X, Theta1, 1)

    # h1 -> output
    [output, z3] = layCalc(a2, Theta2, 0)
    # endregion

    # region - Total Error Calc (non-regularized, we'll add the regularization term after this. For larger datasets this should be vectorized rather than using a for-loop)
    [m,n] = output.shape
    err_total = 0

    for i in range(m):
        outloop = output[i,:]
        subtotal = 0

        # Setting y = 1 when outputs match
        for j in range(n):
            y_out = 0
            if y[i] == j:
                y_out = 1

            error = (-1*y_out)*np.log(outloop[j]) - (1-y_out)*np.log(1-outloop[j])
            subtotal = subtotal + error

        err_total = err_total + subtotal
    J = err_total/m
    # endregion

    # region - Regularization
    Theta1Sum = thetaSum(Theta1)
    Theta2Sum = thetaSum(Theta2)

    J = J + (lam/(2*m))*(Theta1Sum + Theta2Sum)
    # endregion
    return J

# This will calculate the output probabilities. Print the result if you want to compare the Neural Network's first guess with its guesses after training. The input arguments are identical to nnCostFunction
def showFinalOut(nn_params, il, ol, X, y, lam, h1):

    # region Re-constructing Thetas from nn_params
    s1 = (il + 1) * h1
    s2 = (h1 + 1) * ol

    a = np.array(nn_params[0:s1])
    b = np.array(nn_params[s1:s1 + s2])

    Theta1 = np.reshape(a, (h1, il + 1))
    Theta2 = np.reshape(b, (ol, h1 + 1))
    # endregion

    # region Layer Calcs
    m = len(X)
    bias = np.matrix(np.ones((m)))
    X = np.concatenate((bias.T, X), 1)

    # input -> h1
    [a2, z2] = layCalc(X, Theta1, 1)

    # h1 _> output
    [output, z3] = layCalc(a2, Theta2, 0)


    return output

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Main Code
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

trainset = np.genfromtxt('trainset.csv', delimiter=",") # load in the training samples
ytrain = np.genfromtxt('ytrain.csv', delimiter=",") # load in the ground truths for the training samples
ytrain = ytrain - 1 # The original labels are "1,2,3" and we want "0,1,2" to make things easier to work with, since Python begins indexing at 0

# Some parameters
il = trainset.shape[1] # ... + 1 bias node
h1 = 5 # number of nodes in the hidden layer
ol = 3 # we're looking at 3 classes, so naturally the output layer will be 3 nodes

# Must have [hidden_layers] + 1 matrices of weights
Theta1 = randInitThetas(il, h1)
Theta2 = randInitThetas(h1, ol)

# Weights must be formatted as a column vector to optimize
a = np.array([ Theta1.ravel() ]).T
b = np.array([ Theta2.ravel() ]).T
initial_nn_params = np.concatenate((a,b))


lam = 0 # don't really need regulization it seems
par = (il, ol, trainset, ytrain, lam, h1) # a tuple of the arguments, just so we don't have type so much when we pass it to functions
opt = {'maxiter': 100} # 100 iterations seems to work just fine for this dataset

# The optimization function from scipy. Uses conjugate gradient to minimize, and automatically implements back-propogation for the given cost function to minimize J by adjusting the theta weights that it is given.
optim = minimize(nnCostFunction, initial_nn_params, method = 'CG', args = par, options = opt)

finalthetas = np.matrix(optim.x).T

final_out = showFinalOut(finalthetas, il, ol, trainset, ytrain, lam, h1)
initial_out = showFinalOut(initial_nn_params, il, ol, trainset, ytrain, lam, h1)

print("\n The probabilities corresponding to classes 1, 2, or 3 BEFORE training: \n")
dprint(initial_out)

print("\n The probabilities corresponding to classes 1, 2, or 3 AFTER training: \n")
dprint(final_out)

# determining accuracy
testset = np.genfromtxt('testset.csv', delimiter=",")
testset = testset[:,0:2]
y_test = np.genfromtxt('ytest.csv', delimiter=",")
y_test = y_test-1

test_predictions = showFinalOut(finalthetas, il, ol, testset, y_test, lam, h1)
test_predictions = np.array(np.argmax(test_predictions, 1)).T
right = sum(test_predictions == y_test) / len(test_predictions) * 100
right = np.around(right, 2)

print("\nAccuracy of the model (should be between 95-100% since data for this set is linearly seperable) :", str(right))
print("\n\nIf you're taking a look at my code on GitHub and running it to see the results for yourself, and if you are new to Neural Networks, I recommend scrolling up and taking a look at the output probabilities before and after training for classes 1 (col 1), 2 (col 2), and 3 (col 3). You'll notice that at first the network doesn't seem to have any idea what is what, and will likely just group everything into the same class (according to the highest probability). But, once the network is trained, the highest probability in each row should correspond with the column of the class that it belongs to.")
