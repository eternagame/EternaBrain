![EternaBrain](https://github.com/EteRNAgame/EternaBrain/blob/master/eternabrain_logo.png)
# EternaBrain

[![Release](https://img.shields.io/badge/release-v1.1-brightgreen.svg)](https://github.com/EteRNAgame/EternaBrain/releases)
[![Python27](https://img.shields.io/badge/python-2.7-blue.svg)](https://www.python.org/download/releases/2.7/)
[![License](https://img.shields.io/badge/license-LGPL--2.1-black.svg)](https://github.com/EteRNAgame/EternaBrain/blob/master/LICENSE)

Using [Eterna](http://eternagame.org) data to understand and predict how players solve RNA folding puzzles.
* These data are move sets graciously donated by Eterna players to accelerate scientific research into RNA design
* Neural networks to learn how top players solve Eterna players and to predict solutions to RNA folding puzzles
* Unsupervised learning to group Eterna players based on their style of solving RNA folding puzzles

## Author
[Rohan Koodli](https://github.com/rk900)

## Benchmarks
### [Eterna100](https://daslab.stanford.edu/site_data/pub_pdf/2016_Anderson-Lee_JMB.pdf)
61/100

## Dependencies
Python: `numpy, tensorflow, pandas, seaborn, matplotlib, scikit-learn`

Conda: `viennarna` (run `conda install -c viennarna`, you should be able to run `python -c "import RNA"` without any errors.

`RNAfold` version `1.8.5` from ViennaRNA (see [config-ViennaRNA.md](https://github.com/EteRNAgame/EternaBrain/blob/master/config-ViennaRNA.md) for installation instructions)

R: `ggplot2, reshape2`

## To Use
### Step 1: Generate the training data 
Following curates a subset of training data "eternamoves-select" which trains an effective CNN move predictor with reasonable test accuracy.

#### Selecting expert solutions
Go to `experts.py` and modify the variables `content` and `uidList`. `content` is the puzzle IDs of the puzzles you want movesets on, and `uidList` is the user ID's of the players you want movesets from. You can either specify these manually, or you can use functions to get them for you. `getPid()` will retrieve all the single state puzzles, and `experience` will retrieve all players with an experience above a certain threshold.

Example:
```python
content = getPid() # all single-state puzzles
uidList = experience(3000) # the top 70 experts, or the top 1 percent of all players
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
content = getPid() # all the single state puzzles
max_moves = 30 # all solutions in under 30 moves
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
Load your model into the appropriate locations for the base predictor and location predictor in `predict_pm.py`. Specify the RNA secondary structure, starting nucleotide sequence, and path to Vienna in `DOT_BRACKET`, `NUCLEOTIDES`, and `path`. Also specify the natural energy and target energy in `current_energy` and `target_energy` (default is 0 kcal).

```python
DOT_BRACKET = '((((....))))'
path = os.getcwd() + './RNAfold'
len_puzzle = len(dot_bracket)
NUCLEOTIDES = 'A'*len_puzzle
ce = 0.0 # current energy
te = 0.0 # target energy
```

You can specify the minimum amount of the puzzle you want the CNN to solve (on its own, it generally cannot solve long puzzles). The amount is calculated by how much of the current structure matches the target structure. Once it reaches the threshold specified or completes the maximum number of moves, the sequence moves to the Single Action Playout (SAP), which runs a Monte Carlo Tree Search to determine what mutations bring the RNA molecule closer to the target secondary structure.
```python
MIN_THRESHOLD = 0.65
```

Now, you can run the model and it will attempt to find a nucleotide sequence that will fold into the secondary structure provided.

## Using a pretrained model
Go to `predict_pm.py` and change the value of `DOT_BRACKET` to the desired target RNA structure in dot-bracket notation. Configure RNAfold, and enter the correct path to Vienna 1.8.5 in the `path` field, then run `predict_pm.py`.

## Key Puzzles
### Multi-state puzzles
6892343 - 6892348, 7254756 - 7254761

## Key Players
8627, 55836, 231387, 42833
