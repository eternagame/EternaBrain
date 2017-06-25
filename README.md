# EternaBrain
Using Eterna data to understand and predict how players solve RNA folding puzzles.
EternaBrain uses unsupervised machine learning to group Eterna players based on their style of solving RNA folding puzzles, and uses supervised machine learning with neural networks to predict how players will solve RNA folding puzzles.

## Key Puzzles
### Single-state puzzles
6502994 - 6503000
### Multi-state puzzles
6892343 - 6892348, 7254756 - 7254761

## Key Players
8627, 55836, 231387, 42833

## To Use
### To cluster on a certain puzzle
```python
data, users = read_movesets_pid(filepath,pid) # change pid to the specific puzzle
```
This will cluster all the players solutions to this puzzle.
The algorithm will output the number of clusters as well as the players in each cluster.

### To find a specific player's movesets
```python
puzzles, pid = read_movesets_uid(filepath,uid) # change uid to a user's ID
```
This will output all the player's solutions to every Eterna puzzle he/she has solved.

### To find a player's solution(s) to a specific puzzle
```python
puzzles = read_movesets_uid_pid(filepath,uid,pid,df='list') # change uid and pid to user ID or player ID; df is to display the movesets in either a list or dataframe format
```
This will output the a player's solutions to a specified puzzle.
