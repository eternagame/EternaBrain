# RNA-Prediction
Using Eterna data to predict RNA folds
## Data
### epicfalcon.txt (my solutions)
- solution id (sol_id)
- puzzle id (pid)
- user id (uid)
- moveset columns
  - elapsed
  - begin_from
    - starting sequence
  - moves
    - list of dictionaries containing location and base

### puzzle-structure-data.txt (puzzle information)
- puzzle id (pid)
- structure
  - in RNA dot-bracket notation
- locks
- constraints
  - SHAPE,0
    - target shape
  - [base pair],[number]
    - example: GU,1
    - the number of those pairs needed to solve the puzzle
  - CONSECUTIVE_[base],[number]
    - example: CONSECUTIVE_G,5
    - the number of consecutive bases needed to solve the puzzle
- folder
  - example: NuPACK, Vienna, Basic
  - the type of folding algorithm used

## Programs
### movesetreader.py
Reads in a file containing movesets with puzzle ID's and returns a list of dicts containing the movesets.

### structure.py
Reads in a file containing puzzle information and returns a pandas dataframe (accessible by headers and row numbers)
containing the data.
