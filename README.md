# Logic Gate Simulator with ANN
Lately I've been fascinated with the idea of neural architectures and the way they enables us to model human brain into computer and train them on complicated tasks that would otherwise be out of reach for traditional programming paradigms. Considering the simulation of logic gates, it is an easy task for traditional programming which can be implemeneted using if else selection statements, but I wanted to build it using artificial neural network. This will simulate how we as humans learn logic. 

Table of Contents
-------------
1. Understanding Neural Architecture
2. Designing the Training Dataset
3. Selection of Weights
4. Barebones empty neural network
5. Implementing forward propagation
6. Backpropagation and learning data
7. Saving the learned weights
8. Testing
9. Analysis of accuracy
                

Understanding the Neural Architecture
-------------
First question arises as to how does neural architecture work. Well, it's best to draw parallels of it from a new born child. How do they learn? They learn by observing their surroundings, for exampe, their parents, or sibling if they have any. Watching with a curious eye as to how do they walk, talk, or do, child's mind starts to adapt its actions to mimic those of the others. 

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/giphy.gif)
> Children are the best immitators in the world.

If we are negligent of the vast underlying complexity of human mind, it becomes easy for us to say that we can replicate human mind in machine code but that would be a far fetched fantasy. 
It can be very motivating to think of such fantasies and form goals that seem unachievable. Science has sought for us ways for achieving these goals by distilling them down into fundamental truths and reason up from there. This is exactly the right way of thinking when it comes to building something close in complexity to neural network. It makes perfect sense to start from fundamental decision makers in the world of compter science, logic gates, and simulate them in a way that is different to the way how it is traditionally done.

https://www.youtube.com/watch?v=aircAruvnKk

The link above will guide you through the basics of how neural networks work and it has been an inspiration for me how he approaches the fundamental truths. Now we will design our inputs and outputs to create a dataset upon which our neural network can be trained to simulate and give correct predictions.

Designing the Training Dataset
-------------
![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/globe.gif)

In the real world, for a child, data is observed, collected, and computed in the most seamless fashion, but here when we are to design something as simple as logic gates, we need to look for ways it would enable us to understand the data and be perfect so that our neural network does not have problems processing it.

When in our exams, we are asked to draw the truth table for logic gate, what information do we seek:

1. Type of logic gate
2. Input states


Since here we are dealing with two input logic gates, our neural network needs to identify six different types of them and then take the inputs to tell us what they compute. We can use 3 bit code to identify a logic gate and 2 successive bits to identify inputs. Therefore our neural network stands at 5 inputs and one output for the answer. Well as far as the middle layer is concerned we use it to extract more detailed features and establish a correleation between the input and output if there exist none. Below is the picture of neural network we are going to design. 

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/neural_net.jpg)
> Shape of the neural network

**Dataset:**


|Input   | Output |   | Input | Output||     Input | Output   |
|--------|--------|---|------ |:------:|---|-------|:--------:|
| 00000  |  0     |   | 01000 | 0      |   | 10000 | 0        |
| 00001  |  1     |   | 01001 | 0      |   | 10001 | 1        |
| 00010  |  1     |   | 01010 | 0      |   | 10010 | 1        |
| 00011  |  1     |   | 01011 | 1      |   | 10011 | 0        |
| 00100  |  1     |   | 01100 | 1      |   | 10100 | 1        |
| 00101  |  0     |   | 01100 | 1      |   | 10101 | 0        |
| 00110  |  0     |   | 01100 | 1      |   | 10110 | 0        |
| 00111  |  0     |   | 01100 | 0      |   | 10111 | 1        |

First three bits correspond to each different gate. Here is a table that gives that information:

|Input   | Gate   |  
|--------|--------|
| 000    |  OR    |
| 001    |  NOR   |
| 010    |  AND   |
| 011    |  NAND  |
| 100    |  XOR   |
| 101    |  NXOR  |

Selecting the Weights
-------------
If you go through the books and ask any professional data scientist, he will tell you that there is an art to selecting the correct inital weights, and they are very correct on that. The reason for carefully adjusting initail weights is that it can

* negatively affect the training process by contributing to the vanishing or exploding gradient problem
* cause the error function to never find its global minimum
* cause slower weight update and converge slowly

But having a beginners mind and to keep things simple I will go ahead and setup weights matrix randomly, which works more than perfectly for this application. We will later generlize the internal code to adapt to various dimensions of neural network. I use the following code in python to initialize them

```python
import numpy as np

np.random.seed(1)

weights_ih = 2 * np.random.random((5, 10)) - 1
weights_ho = 2 * np.random.random((10, 1)) - 1

```

Barebones of Empty ANN
-------------

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/barebones.png)

Here we create a very basic skeleton for our logical brain with a class containing all functions related to the neural network. It's a good idea to keep everything well organized in order to avoid mess when programs get large.

Initiation:
-------------

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/__init__.png)

First we print out some welcome statements to indicate execution of the constructor. Then we take in the external parameters and assign them to the variables that will be used used inside the class. A learning rate is setup that helps us with the approaching of the global minimum incremently. Weights are randomly setup inside the constructor to enable them at the start of program.

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/training_data.png)
![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/training_targets.png)

We replicate the dataset using numpy arrays, which helps us with fast matrix computations, with the same structure as that discussed in earlier sections. We represent bits with zeros and ones, but they do not give a very accuracte prediction and learning rates, therefore we decide to change them to floating points.This increases the neural network's learning ability by a factor of 10. Small tweaks can lead to optimum parameters.

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/mod_inputs.png)



Implementing Forward Propagation
-------------

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/forward_propagation.png)

We take the inputs from the user and feed them through the input neurons and perform dot product with the weight matrix to evolve a vector which comes out of the hidden layer. Second half of the feed-forward procedure happens in the third line where our outputs are dervied. These can either be used for backpropagtion or for the sole purpose of forward-feed.

Backpropagation
-------------

This architecture, being the most basic of all, is quite efficent for this application. So first we feed forward to acquire results of the two layers. Then find the error at the final layers using error function. Then using the derivative of the error function, deltas, we calculate the deltas for the hidden layer. Then final_delta and hidden_delta help us update the weight matrices, which enables us to learn. 

This function performs only one pass through the neural network updating all the weights and calculating cumulative error at the output node.

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/backprop1.png)

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/backprop.png)




Learning Repeatedly for Perfection in Prediction
-------------

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/train.png)

This function takes in the number of times the model needs to be trained on the training dataset. Usually this neural network take 10 epochs to train to give us accurate prediction. Error is exported to the screen after each epoch. Indexes for the datapoints inside the dataset are provided to the backpropagation function. Higher epoch enable us error that approaches 0, therefore enabling this neural network to give 100% accurate predictions.


Activation Function
-------------
Simply put, the activation function calculates a weighted sum of its input, adds a bias and then decides wether it should be fired or not. We use Rectified Linear Units, or in short ReLU functions, as activation functions. It only fires when the input to the neuron is positive, otherwise throughputs a zero. 
Simply put, it calculates a “weighted sum” of its input, adds a bias and then decides whether it should be “fired” or not 

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/relu.png)
![](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2018/10/Line-Plot-of-Rectified-Linear-Activation-for-Negative-and-Positive-Inputs.png)

The Main Function
-------------
We define the matrix containers for the training datasets and targets, and then we initilize the nural network. It is here we design a command interface with a menu that enables the user to interact with the simulation. They can have the option to train it multiple times with variable parameters. Upon training, they can test the accuracy of predictions, and they can exit the program whenever they like.

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/main.png)
![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/main2.png)
![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/main3.png)

Testing
-------------

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/Peek%202020-03-31%2002-11.gif)

Upon execution and running backpropagation training algorithm, we observe that the error approaches zero rapidly. Therefore, it idicates that there is a strong correlation of inputs with the outputs and neural network is converging, which is a strong indicator of potential predictive model. Above all it works flawlessly!

In the pictures below we see the weights matrix initialization has been altered. While sifting through some research papers I came across some valuable insights on improving the rate of learning by adjusting the weights matrix. Therefore, applying formula to the orignal matrix helped neural network converge. 
![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/analysis_preweight.png)
![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/analysis_postweight.png)

Final Code
-------------

This took me one entire day to write and this is already exhilirating. This is just a small glimpse of what neural networks are capable of. I believe future projects will grow by leaps and bounds as I dive in deeper to refine my techniques in using my own tools.

![](https://github.com/harisrab/logicSim_ANN/blob/master/examples/code.gif)
