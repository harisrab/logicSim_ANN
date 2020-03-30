#Import Libraries
import numpy as np
from numpy import save


np.random.seed(1)

#Class
class neural_function:
    """This neural network simulates logic gates"""

    def __init__(self, training_dataset, training_targets, lr):
        """This initiates the neural network"""
        
        print '[+] Logic Simulator Initiated [###############################]'

        self.training_dataset = training_dataset
        self.training_targets = training_targets
        
        self.learning_rate = lr
        self.error = 0

        #Initilize weights when initializing the neural network
        self.weights_ih = (2 * np.random.random((5, 10)) - 1) * np.sqrt(2.0/5.0)
        self.weights_ho = (2 * np.random.random((10, 1)) - 1) * np.sqrt(2.0/10.0)
        
        
        print '[+] Parameters established [##################################]'
        print '...............................................................'

    def forward_propagation(self, data_vector):
        """Function performs forward propagation"""

        initial_output = data_vector
        hidden_output = self.RELU(np.dot(initial_output, self.weights_ih))
        final_output = np.dot(hidden_output, self.weights_ho)

        return initial_output, hidden_output, final_output

        

    def backpropagation(self, datapoint):
        """Function performs backpropagation"""
        
        #Forward Propagate to find outputs
        init_output, hid_output, fin_output = self.forward_propagation(self.training_dataset[datapoint : datapoint + 1])
        
        #Find Errors
        self.error += np.sum((self.training_targets[datapoint : datapoint + 1] - fin_output) ** 2)

        #Discover Deltas
        final_delta  = (self.training_targets[datapoint : datapoint + 1] - fin_output)
        hidden_delta =  final_delta.dot(self.weights_ho.T) * self.RELU2DERIV(hid_output)

        #Update the weights
        self.weights_ho += self.learning_rate * hid_output.T.dot(final_delta)
        self.weights_ih += self.learning_rate * init_output.T.dot(hidden_delta)
        
        #Print weights and updates
        #print 'Forward Prop Result = ', fin_output
        #print 'Total Error = ', self.error
        #print ''
    

    def train(self, epoch):
        """Function perfroms training given epochs"""
        
        for i in range(epoch):
            print ''
            print '.............Epoch %s ..............' %i
            print ''

            self.error = 0
            
            for datapoint in range(len(self.training_dataset)):               
                self.backpropagation(datapoint)

            print '[+] Error = ', self.error 
        print ''
        print '[-----------Training Complete---------]'

    def RELU(self, x):
        """Activation Function"""
        
        return (x > 0) * x

    def RELU2DERIV(self, x):
        """Derivative for the RELU function"""
        
        return (x > 0)
    
    def get_weights(self):
        return self.weights_ih, self.weights_ho

def convert_input(vector):
    for i in range(len(vector)):
        if vector[i] == '1':
            vector[i] = 0.9

        elif vector[i] == '0':
            vector[i] = 0.1

    return vector

#Main Function
def main():
    """Main Function"""

    training_data = np.array([ [0.1, 0.1, 0.1, 0.1, 0.1],
                               [0.1, 0.1, 0.1, 0.1, 0.9],
                               [0.1, 0.1, 0.1, 0.9, 0.1],
                               [0.1, 0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.1, 0.1],
                               [0.1, 0.1, 0.9, 0.1, 0.9],
                               [0.1, 0.1, 0.9, 0.9, 0.1], 
                               [0.1, 0.1, 0.9, 0.9, 0.9], 
                               [0.1, 0.9, 0.1, 0.1, 0.1], 
                               [0.1, 0.9, 0.1, 0.1, 0.9], 
                               [0.1, 0.9, 0.1, 0.9, 0.1], 
                               [0.1, 0.9, 0.1, 0.9, 0.9], 
                               [0.1, 0.9, 0.9, 0.1, 0.1],
                               [0.1, 0.9, 0.9, 0.1, 0.9],
                               [0.1, 0.9, 0.9, 0.9, 0.1],
                               [0.1, 0.9, 0.9, 0.9, 0.9],
                               [0.9, 0.1, 0.1, 0.1, 0.1],
                               [0.9, 0.1, 0.1, 0.1, 0.9],
                               [0.9, 0.1, 0.1, 0.9, 0.1],
                               [0.9, 0.1, 0.1, 0.9, 0.9],
                               [0.9, 0.1, 0.9, 0.1, 0.1],
                               [0.9, 0.1, 0.9, 0.1, 0.9],
                               [0.9, 0.1, 0.9, 0.9, 0.1],
                               [0.9, 0.1, 0.9, 0.9, 0.9] ]) 
    
    training_targets = np.array([[0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.9]]).T



    learning_rate = 0.02

    #Initiate
    ANN = neural_function(training_data, training_targets, learning_rate)
    
    while True:
        print ''
        print '[+] Select options for Logic Simulator: \n'
        print '1. Train neural network'
        print '2. Use Logic Simulator'
        print '3. Exit'
        print ''

        option = int(input('> Enter your option: '))

        if (option == 1):
            
            epochs = int(input('> Enter Epochs (range 100,000 - 10,000,000): '))
            ANN.train(epochs)
            learned_weights_ih, learned_weights_ho = ANN.get_weights()
            
            #Save the files after training
            save('w_input_hidden.npy', learned_weights_ih)
            save('w_hidden_output.npy', learned_weights_ho)

        elif (option == 2):
            
            gate_to_use = str(raw_input('> Enter the gate to simulate: '))
            gate_to_use = gate_to_use.lower()

            inputs = raw_input('> Enter inputs (format: 1 1): ')
            
            if gate_to_use == "or":
                ANN_input = "000" + inputs[0] + inputs[2]
                ANN_input = np.array(map(int, ANN_input))
                answer = abs(ANN.forward_propagation(convert_input(ANN_input))[2][0])
            
            elif gate_to_use == "nor":
                print 'inputs[0] = ', inputs[0]
                print 'inputs[2] = ', inputs[0]

                ANN_input = "001" + inputs[0] + inputs[2]
                ANN_input = np.array(map(int, ANN_input))
                answer = abs(ANN.forward_propagation(convert_input(ANN_input))[2][0])

            elif gate_to_use == "and":
                ANN_input = "010" + inputs[0] + inputs[2]
                ANN_input = np.array(map(int, ANN_input))
                answer = abs(ANN.forward_propagation(convert_input(ANN_input))[2][0])

            elif gate_to_use == "nand":
                ANN_input = "011" + inputs[0] + inputs[2]
                ANN_input = np.array(map(int, ANN_input))
                answer = abs(ANN.forward_propagation(convert_input(ANN_input))[2][0])

            elif gate_to_use == "xor":
                ANN_input = "100" + inputs[0] + inputs[2]
                ANN_input = np.array(map(int, ANN_input))
                answer = ANN.forward_propagation(convert_input(ANN_input))[2][0]

            elif gate_to_use == "nxor":
                ANN_input = "101" + inputs[0] + inputs[2]
                ANN_input = np.array(map(int, ANN_input))
                answer = ANN.forward_propagation(convert_input(ANN_input))[2][0]
            
            print ''
            print '#######################'
        
            if answer > 0.5:
                print '[+] Answer is True '

            elif answer < 0.5:
                print '[+] Answer is False'
            
            print '#######################\n'
        elif (option == 3):
            break

#Initiation of Main Function
if __name__ == '__main__':
    main()




