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
- Keras is just magic, This will save you ton of time: https://keras.io/getting-started/faq/
- These tutorials will walk you through the basics of tensorflow:
  Youtube channel: https://www.youtube.com/watch?v=wuo4JdG3SvU&list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ
  Git repo with code: https://github.com/Hvass-Labs/TensorFlow-Tutorials/
- Tenorboard is fantastic in visualizing model and the accuracy: https://www.tensorflow.org/get_started/summaries_and_tensorboard

# Games and Data
The selection of the games depended on how much data was behind each game and the developers familiarity of the games. Connect4, Chinese checkers 6 player and breakthrough were select.

The data is from a mysql database [Citation Needed] which contains thousands of matches on many different games. Here is the SQL code used to find the games with the most data. 
```sql
select g.name, count(m.match_id) MatchCount, sum(temp.step) as StepCount,  Max(temp.roleindex) from games g
inner join matches m on m.game = g.name
inner join 
    (
        select match_id, max(step_number) as step, max(roleindex) as roleindex
        from moves 
        group by match_id
    ) temp on temp.match_id = m.match_id
where m.match_id not in (select match_id from errormessages)
group by g.name
order by StepCount desc
```  
Output:
```
   Game Name                       Match Count     Step Count      Num Roles
1  breakthrough                    2786            138626          1
2  connect4                        2355            47266           1
3  numbertictactoe                 1630            11956           1
4  tictactoe                       1405            10350           1
5  tictactoe_3d_small_6player      1291            33204           5
6  tictactoe_3d_6player            1276            74760           5
7  chinesecheckers6                1268            72210           5
8  chinesecheckers6-simultaneous   1174            20007           5
```

## Connect4 
Connect four is a two player turn taking game that is the simplest game that was examined, see https://en.wikipedia.org/wiki/Connect_Four

![Connect four gif](https://upload.wikimedia.org/wikipedia/commons/a/ad/Connect_Four.gif)

### State

The state for connect four is represented with 135 bits, the board is 7x7 and each cell can be red, blue or blank and there are two more bits for marking hows turns it is. Sample state:

> ((CELL 1 1 W) (CELL 1 0 DIRT) (CELL 2 0 DIRT) (CELL 3 0 DIRT) (CELL 4 0 DIRT) (CELL 5 0 DIRT) (CELL 6 0 DIRT) (CELL 7 0 DIRT) (CELL 2 1 B) (CELL 2 2 B) (CELL 2 3 B) (CELL 2 4 B) (CELL 2 5 B) (CELL 2 6 B) (CELL 3 1 B) (CELL 3 2 B) (CELL 3 3 B) (CELL 3 4 B) (CELL 3 5 B) (CELL 3 6 B) (CELL 4 1 B) (CELL 4 2 B) (CELL 4 3 B) (CELL 4 4 B) (CELL 4 5 B) (CELL 4 6 B) (CELL 5 1 B) (CELL 5 2 B) (CELL 5 3 B) (CELL 5 4 B) (CELL 5 5 B) (CELL 5 6 B) (CELL 6 1 B) (CELL 6 2 B) (CELL 6 3 B) (CELL 6 4 B) (CELL 6 5 B) (CELL 6 6 B) (CELL 7 1 B) (CELL 7 2 B) (CELL 7 3 B) (CELL 7 4 B) (CELL 7 5 B) (CELL 7 6 B) (CELL 1 2 B) (CELL 1 3 B) (CELL 1 4 B) (CELL 1 5 B) (CELL 1 6 B) (CONTROL RED) )


### Action
The number of actions is seven, one for each pillar which the tokens be dropped. 

## Chinese checkers
Chinese checkers is a startegy board game where up to 6 players compete, see https://en.wikipedia.org/wiki/Chinese_checkers.

<img src="https://upload.wikimedia.org/wikipedia/commons/3/3e/ChineseCheckersboard.jpeg" alt="Chinese checkers" width="320">

The six player version of chinese checkers was used and the state is represented with 253 bits. This is much more complex than connect four, see sample state:

> ((STEP 14) (CELL C4 RED) (CELL B1 BLANK) (CELL E4 MAGENTA) (CELL C2 BLANK) (CELL F2 BLUE) (CELL F1 BLANK) (CELL F3 TEAL) (CELL H2 BLANK) (CELL F5 GREEN) (CELL F6 BLANK) (CELL C5 YELLOW) (CELL C6 BLANK) (CELL D4 RED) (CELL E1 MAGENTA) (CELL C1 BLANK) (CELL G5 BLUE) (CELL G1 BLANK) (CELL G4 TEAL) (CELL H1 BLANK) (CELL E5 GREEN) (CELL G7 BLANK) (CELL C3 YELLOW) (CELL C7 BLANK) (CELL B2 BLANK) (CELL A1 RED) (CELL D1 MAGENTA) (CELL D2 BLANK) (CELL D3 BLANK) (CELL D5 BLANK) (CELL D6 YELLOW) (CELL E2 BLANK) (CELL E3 BLANK) (CELL F4 BLANK) (CELL G2 BLUE) (CELL G3 BLANK) (CELL G6 GREEN) (CELL I1 TEAL) (CONTROL YELLOW) )

There are in total 390 actions that can be performed in this game however each role can only perform 90 of them. E.g role1 is the only role able to perform (MOVE A1 B1). However only a few are leagal actions in any given state. 

## Breakthrough

Breakthrough is a two player turn taking game, basicly chess with only pawns , see https://en.wikipedia.org/wiki/Breakthrough_(board_game). 

<img src="https://raw.githubusercontent.com/thorgeirk11/IndepententStudy/master/screenshots/breakthrough_wikipedia.png" alt="Chinese checkers" width="320">


# Training data
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

# Thoughts while developing
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


## Ideas for further development
- Shuffle the state and action representations.
    To reuse the same data we could shuffle the bit array and train the model on many diffrent state and action representations for a single game. This could force the network to learn the lower and higher represntation of the game in the aproprate layers and move the logic resoning to the middle layer. 

- Training with frozen middle layer.
    A interesting place for further research would be to se if training on multiple roles and then frezing the middle layer results in better performance when training on a unseen role.


# Design of the model

As described in the problems description, the task is to design a ANN that can train on multiple games where the middle part of the network trains on every game and the input and output layers get changed between games. 

All the games described in GDL are markov decision processes, meaning that at each state of the game it does not matter how you ended up in that state, the current state only matters (https://en.wikipedia.org/wiki/Markov_property). Therefore, I decided not to invest time into LSTMs or RNNs for this problem, however it would be interesting to see if training an LSTM on these games would improve the accuracy since it would learn the play style of the agents.

I designed the system to use fully connected layers and to start off a single game.

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
Snapshot from tensorboard of the model:

<img src="https://raw.githubusercontent.com/thorgeirk11/IndepententStudy/master/screenshots/single_role.png" width="400">

## Multiple role model
Since we want to split up the learning between roles for the games the next step was to create a model that could handle 
Here the model was expanded to allow for multiple output while only taking a single input. The state representation did not between roles only the output space.

```py
with tf.name_scope("input_network"):
    in_con4 = Input(shape=(135,))
    con4 = Dense(200, activation='relu')(in_con4)

with tf.name_scope("middle_network"):
    mid = Dense(500, activation='relu')(con4)
    mid = Dropout(0.5)(mid)
    mid = Dense(500, activation='relu')(mid)

models = []
for i in range(2):
    with tf.name_scope("output_network"):
        out = Dense(50, activation='relu')(mid)
        out = Dense(8, activation='softmax')(out) 
        models.append(Model(inputs=in_con4, outputs=out))

# models = [role0, role1]
```
Snapshot from tensorboard of the model:

<img src="https://raw.githubusercontent.com/thorgeirk11/IndepententStudy/master/screenshots/multiple_roles_expanded.png" width="400">


## Multiple game model

This model the most advanced model 

```py
with tf.name_scope("input_network_connect4"):
    in_con4 = Input(shape=(135,))
    con4 = Dense(200, activation='relu')(in_con4)

with tf.name_scope("input_network_breakthrough"):
    in_bt = Input(shape=(130,))
    bt = Dense(200, activation='relu')(in_bt)

with tf.name_scope("middle_network"):
    middle = Sequential([
        Dense(500, activation='relu', input_shape=(200,)),
        Dropout(0.5),
        Dense(500, activation='relu')
    ])
    con4_mid = middle(con4)
    bt_mid = middle(bt)

con4_models = []
for i in range(2):
    with tf.name_scope("output_network_conncet4_role{0}".format(i)):
        out = Dense(50, activation='relu')(con4_mid)
        out = Dense(8, activation='softmax')(out)
        model = (Model(inputs=in_con4, outputs=out), "connect4", i)
        con4_models.append(model)

bt_models = []
for i in range(2):
    with tf.name_scope("output_network_breakthrough_role{0}".format(i)):
        out = Dense(200, activation='relu')(bt_mid)
        out = Dense(155, activation='softmax')(out)
        model = (Model(inputs=in_bt, outputs=out), "breakthrough", i)
        bt_models.append(model)
```

Here is a diagram of the model compiled with only connect4 and breakthrough, note that there are two inputs one for each game and four outputs one for each role for both games.

![Model with connect4 and breakthrough](https://raw.githubusercontent.com/thorgeirk11/IndepententStudy/master/screenshots/Connect4_and_breakthrough.png)
