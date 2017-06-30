# Indepentent Study

## Introduction
General game playing (GGP) seeks to create intelligent agents capable of independently playing many different games effectively without knowing the rules of the game beforehand. Humans have the ability of learning a strategy in one game and apply it on another, effectively transferring the knowledge gained between games. This ability has been hard to replicate in GGP agents.  
        
GGP competitions have been held for many years and their matches have been recorded. A match consist of series of actions taken from each agent participating in the game. These datasets contain many different games, some of which have over a hundred thousand recorded matches.  Each match was played by a GGP agent, where the agent tried to intelligently pick an optimal action for each state presented. 
        
Artificial neural networks (ANN) have gotten a lot of traction for their ability to learn from massive amount of raw data and recognize advance patterns.  An ANN is composed of an input layer, hidden layer and an output layer. A general hypothesis is that if an ANN is trained on a game the input layer is used to read in the current state of the game and the output layer is responsible for picking an action based on some learned policy. Where the hidden layer is the layer that learns the general notion of the game, the basic feature extraction and reasoning.
    
## Problem Description
In this study, the author will employ deep learning techniques and use an ANN to learn the probability of an action being performed on a state for a given game. The deep ANN will have multiple layers and will be designed so that input, output and immediate surrounding layers can be switched out between games. Only the middle part of the hidden layers would remain intact between the games. 
        
A game state in the game description language  (GDL) consists of propositions that are either true or false and therefore a state can be treated as a boolean array. This boolean array is the input into the ANN. The output of the ANN would be the probability for every action in the game for that given state.
        
A considerable part of the project is to figure out a good structure for the  network (number and size of hidden layers) and play around with the hyper-parameters.
     

# Development

# Tutorial and Supporting metarials

Here is a list of tutorials which I found really usefull:
- These tutorials will walk you throught the basics of tensorflow:
  Youtube chanel: https://www.youtube.com/watch?v=wuo4JdG3SvU&list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ
  Git repo with code: https://github.com/Hvass-Labs/TensorFlow-Tutorials/
- Tenorboard is fantastic in visulizing model and the accuracy: https://www.tensorflow.org/get_started/summaries_and_tensorboard
- For saving and restoring subgraphs: https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125

## Training data
The data is taken from a GGP database [Citation Needed]. 
The states for the games are extracted from the database using the Parse.sql file located under the sql folder for each game. The Parse.sql file selects the states and the actions performed on that state for each role in the given game. It outputs a .csv file containg the state and actions.

Example line form the training data .csv:
>  Match.connect4.1240881643605,9,0,0,-1,1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,0,0,0,0,0,0,1,0

This example is taken from connect4. Here we see that the first 4 coulums contain some meta data:

1. MatchId, not used by the code but might be usefull,
2. Step index, tells at which step the state is form the start of the match, 
3. Role index, which role dose the action corraspond to.
4. Score, the score for the role in this match. Can be used to filter out only moves made by the winning roles.

Next comes the encoded state, that is there are 135 rules in connect4 and each rule is either true (1) or false (-1). This state represation is used as input into the neural network. 

The last 9 coulums are the actions performed on the state for the given role. This is called one hot encoding (https://www.tensorflow.org/api_docs/python/tf/one_hot), basicly only one of the actions is performed (1) in a given state and the rest are not performed (0).

Note that the actions perfomed can be diffrent between roles, see chinese checkers. In chinese checkers the sql files are six one for each role. 

## Thoguhts while developing
Connect4 was the first game I converted and trained on, thses are the mistakes that I encounted in the beging.

- Inperfect data.
    Always keep in mind that the data is not perfect and often agents perform poorly. This will affect your model and make it harder for it to achive really good accuracy. 

- Amount of data.
    For deep neural networks it is better to have more training data than less. This is the reason Connect4, tictactue3d and chines checkers were selected, they contain the most data. The amount of data is however not great, for example chineses checkers for role0 there are 5892 states, which is not a lot.

- How to messure accuracy. 
    One way to mesure accuarcy is the mesure the  distance between the predicted labels and the true labels, using for example sum of sqare diffrentances.
    Another way is to messure how often the network prediced the correct label. The only problem with this approch is that if for example two actions are equaly likly on a given state then the network will only achive 50% accuracy on that state.

- Split up the roles. 
    That is do not train on the model on both roles with out separation, since the stratigy for each role is diffrent and the network could get confused.

# Initial desgin

As described in the problems describition, the task is to desgin a ANN that can train on multiple games where the middle part of the network trains on every game and the input and output layers get changed. 

All the games described in GDL are markov decision processes, meaning that at each state of the game it dose not matter how you ended up in that state, the current state only matters (https://en.wikipedia.org/wiki/Markov_property). Therefore I descided not to invest in research into LSTMs or RNNs for this problem, however it would be interesting to see if training an LSTM on these games would improved the accuracy since it would learn the play style of the agents.

I desgined the system to use fully connceted layers and experimented with many hyper paramiters.

To start off a single game was used and  tryed out many diffrent variations for networks. 


```py
state_size = 135
num_actions = 8

# Input into the network, the state
input_state = tf.placeholder(tf.float32, shape=[None, state_size], name='input_state')
# The actual action performed in the state (label)
y_true = tf.placeholder(tf.float32, shape=[None, num_actions], name='label')
y_true_cls = tf.argmax(y_true, dimension=1)

input_state_pretty = pt.wrap(input_state)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    pred_action, _ = input_state_pretty.\
        fully_connected(size=128, name='layer_fc2').\
        fully_connected(size=64,  name='layer_fc3').\
        fully_connected(size=128, name='layer_fc4').\
        fully_connected(size=64,  name='layer_fc5').\
        fully_connected(size=128, name='layer_fc6').\
        softmax_classifier(num_classes=num_actions, labels=y_true)
loss = tf.reduce_mean(tf.squared_difference(pred_action, y_true))

y_pred_cls = tf.argmax(pred_action, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
```

This is the initial setup of the network, 






# Initial results 


