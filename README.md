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
- read_movesets
  - Reads in a file containing movesets data and returns a list of dicts containing the movesets.
- puzzle_attributes
  - Reads in a file containing movesets data and returns a list of the solution id, puzzle id, or user id, depending on what the
  user specifies in the function.

### structure.py
- read_structure
  - Reads in a file containing puzzle information and returns a pandas dataframe (accessible by headers and row numbers)
containing the data.

### getData.py
- getData_pid
  - Returns the movesets and structure given the pid, pid list (from the movesets), movesets, and puzzle structure.
