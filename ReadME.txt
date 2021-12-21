To run the CNN run the run_cnn.py file
To change the Dropout Rate modify the dropout variable
To use Droput set use_dropout to 1
To not use Droput set use_dropout to 0
To change Activation Function modify the activation_function variable
To change Learning Rate modify learning_rate variable

# to use ReLU
activation_function = nn.ReLU()
# to use Tanh
activation_function = nn.Tanh()
# to use ELU
activation_function = nn.ELU()
# to use Sigmoid
activation_function = nn.Sigmoid()

Cross Entropy loss function already uses a softmax activation function
so an activation function isn't used on the output layer

Lab7 for base implementation Then modified myself
Youtube video to help initialize weights reference below:
Aladdin Persson, “Youtube,” 10 April 2020. [Online]. Available: https://www.youtube.com/watch?v=xWQ-p_o0Uik.