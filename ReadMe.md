# Indipendant Study

## Introduction
General game playing (GGP) seeks to create intelligent agents capable of independently playing many different games effectively without knowing the rules of the game beforehand. Humans have the ability of learning a strategy in one game and apply it on another, effectively transferring the knowledge gained between games. This ability has been hard to replicate in GGP agents.  
        
GGP competitions have been held for many years and their matches have been recorded. A match consist of series of actions taken from each agent participating in the game. These datasets contain many different games, some of which have over a hundred thousand recorded matches.  Each match was played by a GGP agent, where the agent tried to intelligently pick an optimal action for each state presented. 
        
Artificial neural networks (ANN) have gotten a lot of traction for their ability to learn from massive amount of raw data and recognize advance patterns.  An ANN is composed of an input layer, hidden layer and an output layer. A general hypothesis is that if an ANN is trained on a game the input layer is used to read in the current state of the game and the output layer is responsible for picking an action based on some learned policy. Where the hidden layer is the layer that learns the general notion of the game, the basic feature extraction and reasoning.
    
## Problem Description
In this study, the author will employ deep learning techniques and use an ANN to learn the probability of an action being performed on a state for a given game. The deep ANN will have multiple layers and will be designed so that input, output and immediate surrounding layers can be switched out between games. Only the middle part of the hidden layers would remain intact between the games. 
        
A game state in the game description language  (GDL) consists of propositions that are either true or false and therefore a state can be treated as a boolean array. This boolean array is the input into the ANN. The output of the ANN would be the probability for every action in the game for that given state.
        
A considerable part of the project is to figure out a good structure for the  network (number and size of hidden layers) and play around with the hyper-parameters.
     

# Training data



# Connect4 

## Extracting data:
The sql scripts for extracting moves and states are located under the sql folder:
### Parse.sql 
    This 

1. Do not aggegate the output of training.
    - Maby just copy paste..

2. Split up the output of diffrent roles.. 
   - have 14 actions for connect4

3. Once you get somthing working split up the output layer for diffrent roles.

4. Use somthing like SSD sum of sqare diffrentances 
    Look at alpha go! :)  
    - Sum ( o(x) - y ) ^ 2

 
