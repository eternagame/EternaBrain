import os

file = os.getcwd() + '\movesets\epicfalcon.txt'

with open(file) as epicfalcon:
    epicfalcon = epicfalcon.readlines()

print (epicfalcon)


