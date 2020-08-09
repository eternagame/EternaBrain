import pandas as pd
from tqdm import tqdm
from predict_pm import predict
import os

def predict_puzzles(puzzle_list, vienna_version=1):
    p = 0
    if vienna_version == 1:
        p = pd.read_csv(os.getcwd()+'/movesets/eterna100.txt', sep=' ', header='infer', delimiter='\t')
    else:
        p = pd.read_csv(os.getcwd()+'/movesets/eterna100_vienna2.txt', sep=' ', header='infer', delimiter='\t')

    solves = []

    for pid in puzzle_list:
        print(pid)
        print('\n\n\n\n\n\n')
        struc = p.iloc[pid, ]['Secondary Structure']
        
        num = p.iloc[pid, ]['Puzzle #']
        name = p.iloc[pid, ]['Puzzle Name']
        print(name, struc)
        try:
            solved, solution = predict(struc, vienna_version)
        except TypeError:
            solution = 'a'
            solved = True
        # WRITE PID, PUZZLE NAME, STRUC, SOLUTION TO FILE
        with open(os.getcwd()+'/movesets/puzzle_solutions_v%i.txt' % vienna_version, 'a') as f:
            f.write('%i\t%s\t%s%s\n' % (num, name, struc, solution))
        
        with open(os.getcwd()+'/movesets/puzzle_results_v%i.txt' % vienna_version, 'a') as f:
            if solved:
                f.write('%i\n' % 1)
            else:
                f.write('%i\n' % 0)

        if solved:
            solves.append(1)
        else:
            solves.append(0)
        
    print(sum(solves))
    return solves

if __name__ == "__main__":
    l = list(range(100))
    predict_puzzles(l, 2)
