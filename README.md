# RNA-Prediction
Using Eterna data to understand how players solve RNA folding puzzles

## Data
### epicfalcon.txt (my solutions) & move-set-11-14-2016.txt (all recorded solutions to puzzles)
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
### readData.py
- read_movesets
  - Reads in a file containing movesets data and returns a list of dicts containing the movesets.
- puzzle_attributes
  - Reads in a file containing movesets data and returns a list of the solution id, puzzle id, or user id, depending on what the
    user specifies in the function.
- read_structure
  - Reads in a file containing puzzle information and returns a pandas dataframe (accessible by headers and row numbers)
    containing the data.

### getData.py
- getData_pid
  - Returns the movesets and structure given the pid, pid list (from the movesets), movesets, and puzzle structure.
  
### encodeRNA.py
- encode_structure
  - Takes an RNA structure in dot-bracket notation and returns it encoded (0 for unpaired base, 1 for paired base).
- encode_movesets
  - Takes moveset data and returns a list containing each move with base (A = 1, U = 2, G = 3, C = 4) and location.
