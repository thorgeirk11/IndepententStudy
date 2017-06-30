# Connect4 
Connect four is a two player turn taking game that is the simplest game that was examined. See https://en.wikipedia.org/wiki/Connect_Four

![](https://upload.wikimedia.org/wikipedia/commons/a/ad/Connect_Four.gif)


## State Represntation 

A game state in the game description language  (GDL) consists of propositions that are either true or false and therefore a state can be treated as a boolean array. This boolean array is the input into the ANN. However since we normalize our input data the array constist of 1 and -1.

The output of the ANN is the probability for every action in the game for that given state.


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

 