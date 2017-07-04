# Indepentent Study

## Introduction
General game playing (GGP) seeks to create intelligent agents capable of independently playing many different games effectively without knowing the rules of the game beforehand. Humans have the ability of learning a strategy in one game and apply it on another, effectively transferring the knowledge gained between games. This ability has been hard to replicate in GGP agents.  
        
GGP competitions have been held for many years and their matches have been recorded. A match consists of series of actions taken from each agent participating in the game. These datasets contain many different games, some of which have over a hundred thousand recorded matches.  Each match was played by a GGP agent, where the agent tried to intelligently pick an optimal action for each state presented. 
        
Artificial neural networks (ANN) have gotten a lot of traction for their ability to learn from massive amount of raw data and recognize advance patterns.  An ANN is composed of an input layer, hidden layer and an output layer. A general hypothesis is that if an ANN is trained on a game the input layer is used to read in the current state of the game and the output layer is responsible for picking an action based on some learned policy. Where the hidden layer is the layer that learns the general notion of the game, the basic feature extraction and reasoning.
    
## Problem Description
In this study, the author will employ deep learning techniques and use an ANN to learn the probability of an action being performed on a state for a given game. The deep ANN will have multiple layers and will be designed so that input, output and immediate surrounding layers can be switched out between games. Only the middle part of the hidden layers would remain intact between the games. 
        
A game state in the game description language (GDL) consists of propositions that are either true or false and therefore a state can be treated as a boolean array. This boolean array is the input into the ANN. The output of the ANN would be the probability for every action in the game for that given state.
        
A considerable part of the project is to figure out a good structure for the network (number and size of hidden layers) and play around with the hyper-parameters.
     


# Tutorial and Supporting materials

Here is a list of tutorials which I found really useful:
- These tutorials will walk you through the basics of tensorflow:
  Youtube channel: https://www.youtube.com/watch?v=wuo4JdG3SvU&list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ
  Git repo with code: https://github.com/Hvass-Labs/TensorFlow-Tutorials/
- Tenorboard is fantastic in visualizing model and the accuracy: https://www.tensorflow.org/get_started/summaries_and_tensorboard
- For saving and restoring subgraphs: https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125

# Games selected

## Connect4 
Connect four is a two player turn taking game that is the simplest game that was examined. See https://en.wikipedia.org/wiki/Connect_Four

![](https://upload.wikimedia.org/wikipedia/commons/a/ad/Connect_Four.gif)


## Training data
The data is taken from a GGP database [Citation Needed]. 
The states for the games are extracted from the database using the Parse.sql file located under the sql folder for each game. The Parse.sql file selects the states and the actions performed on that state for each role in the given game. It outputs a .csv file that contains the state and actions.

Example line form the training data .csv:
>  Match.connect4.1240881643605,9,0,0,-1,1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,0,0,0,0,0,0,1,0

This example is taken from connect4. Here we see that the first 4 columns contain some meta data:

1. MatchId, not used by the code but might be useful,
2. Step index, tells at which step the state is form the start of the match, 
3. Role index, which role dose the action correspond to.
4. Score, the score for the role in this match. Can be used to filter out only moves made by the winning roles.

Next comes the encoded state, that is there are 135 rules in connect4 and each rule is either true (1) or false (-1). This state representation is used as input into the neural network. 

The last 9 columns are the actions performed on the state for the given role. This is called one hot encoding (https://www.tensorflow.org/api_docs/python/tf/one_hot), basically only one of the actions is performed (1) in each state and the rest are not performed (0).

Note that the actions performed can be different between roles, see Chinese checkers. In Chinese checkers, the sql files are six one for each role. 

## Thoughts while developing
Connect4 was the first game I converted and trained on, these are the mistakes that I encountered in the beginning.

- Imperfect data.
    Always keep in mind that the data is not perfect and often agents perform poorly. This will affect your model and make it harder for it to achieve really good accuracy. 

- Amount of data.
    The amount of data is always a factor to consider when training a classifier. Connect4, breakthrough and chines checkers were selected because they contain the most data. The amount of data is however not great, for example chinses checkers for role0 there are 5892 states, which is not a lot.

- How to measure accuracy. 
    One way to measure accuracy is the measure the distance between the predicted labels and the true labels, using for example sum of square differentness.
    Another way is to measure how often the network predicted the correct label. The only problem with this approach is that if for example two actions are equally likely on a given state then the network will only achieve 50% accuracy on that state.

- Split up the roles. 
    That is do not train on the model on both roles without separation, since the strategy for each role is different and the network could get confused.

# Initial design

As described in the problems description, the task is to design a ANN that can train on multiple games where the middle part of the network trains on every game and the input and output layers get changed between games. 

All the games described in GDL are markov decision processes, meaning that at each state of the game it does not matter how you ended up in that state, the current state only matters (https://en.wikipedia.org/wiki/Markov_property). Therefore, I decided not to invest time into LSTMs or RNNs for this problem, however it would be interesting to see if training an LSTM on these games would improve the accuracy since it would learn the play style of the agents.

I designed the system to use fully connected layers and experimented with many hyper parameters.

To start off a single game was used and tried out many different variations for networks. 

This is the initial setup of the network, 

## Single role model
The initial model used, note that it is written in pretty-tensor:

The model itself has five fully connected layers and the model was trained on the game connect4 since the state size is 135 and the number of actions is 8. [Mean squared error]((https://en.wikipedia.org/wiki/Mean_squared_error)) (MSE) is used for the loss function, mainly because it was used in alpha-go   (citation needed) and meaningful representation of loss. MSE has a range between 0, no difference and 1 max distance.   
```py
state_size = 135
num_actions = 8

# Input into the network, the state
x = tf.placeholder(tf.float32, shape=[None, state_size], name='x')
# The actual action performed in the state (label)
y_true = tf.placeholder(tf.float32, shape=[None, num_actions], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

x_pretty = pt.wrap(x)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    pred_action, _ = x_pretty.\
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
Snap shot from tensorboard of the model:
![alt text](https://raw.githubusercontent.com/thorgeirk11/IndepententStudy/master/single_role.png)

## Multiple role model
Here the model was expanded to allow for multiple output while only taking a single input. The state representation did not between roles only the output space.

```py
with tf.name_scope("input_network"):
    in_con4 = Input(shape=(135,))
    con4 = Dense(200, activation='relu')(in_con4)

with tf.name_scope("middle_network"):
    mid = Dense(500, activation='relu', input_shape=(200,))(con4)
    mid = Dropout(0.5)(mid)
    mid = Dense(500, activation='relu')(mid)

with tf.name_scope("output_network"):

models = []
for i in range(2):
    with tf.name_scope("output_network"):
        out = Dense(50, activation='relu')(mid)
        out = Dense(8, activation='softmax')(out) 
        models.append(Model(inputs=in_con4, outputs=out))

# model = [role0, role1]
```

## Multiple game model

This model the most advanced model 

```py
with tf.name_scope("input_network"):
    in_con4 = Input(shape=(135,))
    in_cc6 = Input(shape=(253,))
    in_bt = Input(shape=(130,))

    con4 = Dense(200, activation='relu')(in_con4)
    cc6 = Dense(200, activation='relu')(in_cc6)
    bt = Dense(200, activation='relu')(in_bt)

with tf.name_scope("middle_network"):
    middle = Sequential([
        Dense(500, activation='relu', input_shape=(200,)),
        Dropout(0.5),
        Dense(500, activation='relu')
    ])
    con4_mid = middle(con4)
    cc6_mid = middle(cc6)
    bt_mid = middle(bt)

con4_models = []
for i in range(2):
    with tf.name_scope("output_network"):
        out = Dense(50, activation='relu')(con4_mid)
        out = Dense(8, activation='softmax')(out)
        model = (Model(inputs=in_con4, outputs=out), "connect4", i)
        con4_models.append(model)

cc6_models = []
for i in range(6):
    with tf.name_scope("output_network"):
        out = Dense(200, activation='relu')(cc6_mid)
        out = Dense(90, activation='softmax')(out)
        model = (Model(inputs=in_cc6, outputs=out), "chinese_checkers_6", i)
        cc6_models.append(model)

bt_models = []
for i in range(2):
    with tf.name_scope("output_network"):
        out = Dense(200, activation='relu')(bt_mid)
        out = Dense(155, activation='softmax')(out)
        model = (Model(inputs=in_bt, outputs=out), "breakthrough", i)
        bt_models.append(model)
```

