# EternaBrain
Using Eterna data to understand and predict how players solve RNA folding puzzles.
* Neural networks to learn how top players solve Eterna players and to predict solutions to RNA folding puzzles
* Unsupervised learning to group players based on their style of solving RNA folding puzzles

## Author
Rohan Koodli

## Key Puzzles
### Multi-state puzzles
6892343 - 6892348, 7254756 - 7254761

## Key Players
8627, 55836, 231387, 42833

## Dependencies
Python: numpy, tensorflow, pandas, seaborn, matplotlib

RNAfold from ViennaRNA

EteRNAbot - You will have to clone the EteRNAbot repository and move the files to your directory. Currently, there is no installation option for EteRNAbot.

## To Use
### Step 1: Generate the training data
#### Selecting expert solutions
Go to `experts.py` and modify the variables `content` and `uidList`. `content` is the puzzle ID's of the puzzles you want movesets on, and `uidList` is the user ID's of the players you want movesets from. You can either specify these manually, or you can use functions to get them for you. `getPid()` will retrieve all the single state puzzles, and `experience` will retrieve all players with an experience above a certain threshold. 

Example:
```python
content = getPid() # all single-state puzzles
uidList = experience(3000) # the top 70 experts
```
or, if you want less puzzles and more experts, you can read in `teaching-puzzle-ids.txt`, which contains 92 key puzzles:
```python
with open(os.getcwd()+'/movesets/teaching-puzzle-ids.txt') as f:
    progression = f.readlines()
progression = [x.strip() for x in content]
progression = [int(x) for x in content]
progression.extend([6502966,6502968,6502973,6502976,6502984,6502985,6502993, \
                6502994,6502995,6502996,6502997,6502998,6502999,6503000])
content = progression
uidList = experience(1000)
```

#### Selecting the fastest solutions
Go to `fastest.py` and modify `content` and `max_moves`. `content` requires the same inputs as above, and `max_moves` is an integer specifying the maximum number of moves you want the data to have.

Example:
```python
content = getPid()
max_moves = 30
```

### Step 2: Training the convolutional neural network (CNN)
EternaBrain uses a convolutional neural net (CNN). Run both `baseCNN.py` and `locationCNN.py`. Just specify the path and name of your pickled data files here:
```python
for pid in content:
    try:
        feats = pickle.load(open(os.getcwd()+'/pickles/X-exp-loc-'+str(pid),'rb'))
        ybase = pickle.load(open(os.getcwd()+'/pickles/y-exp-base-'+str(pid),'rb'))
        yloc = pickle.load(open(os.getcwd()+'/pickles/y-exp-loc-'+str(pid),'rb'))
        for i in range(len(feats)):
            feats[i].append(yloc[i])
        real_X.extend(feats)
        real_y.extend(ybase)
        pids.append(feats)
    except IOError:
        continue
```
Specify the name and directory of where you want the model to be saved here:
```python
saver.save(sess, os.getcwd()+'/models/base/baseCNN')
saver.export_meta_graph(os.getcwd()+'/models/base/baseCNN.meta')
```

### Step 3: Predicting
Load your model into the appropriate locations for the base predictor and location predictor in `predict_pm.py`. Specify the RNA secondary structure and starting nucleotide sequence in `dot_bracket` and `nucleotides`, respectively. Also specify the natural energy and target energy in `current_energy` and `target_energy`, respectively (default is 0 kcal). 

```python
dot_bracket = '((((....))))'
len_puzzle = len(dot_bracket)
nucleotides = 'A'*len_puzzle
ce = 0.0 # current energy
te = 0.0 # target energy
```

You can specify the minimum amount of the puzzle you want the CNN to solve (on its own, it generally cannot solve long puzzles). The amount is calculated by how much of the current structure matches the target structure. Once it reaches the threshold specified or completes the maximum number of moves, the sequence moves to the reinforcement learner and the domain specific pipeline. If you want the CNN to completely solve the puzzle, set `min_threshold` to 1.0
```python
min_threshold = 0.65
```

Then, you can run the model and it will attempt to match the secondary structure you specified.
