'''
Plots the test, training, and baseline accuracies for the CNNs
'''
from matplotlib import pyplot as plt
import seaborn; seaborn.set()
import os
import numpy as np

TYPE = 'base'

if TYPE == 'base':
    with open(os.getcwd()+'/baseCNN14.out') as f:
        content = f.readlines()
    t1 = [0.0]*50
    t2 = [0.24]*50 #24
    t3 = [0.42]*50 #42
    t4 = [0.51] #51
    baseline = [0.25]*151
    plt.title('Cross-Validation Accuracy of base predictor',size=14)

elif TYPE == 'location':
    with open(os.getcwd()+'/locationCNN14.out') as f:
        content = f.readlines()
    t1 = [0.0]*50
    t2 = [0.15]*50 #24
    t3 = [0.25]*50 #42
    t4 = [0.34] #51
    baseline = [0.02]*151
    plt.title('Cross-Validation Accuracy of location predictor',size=14)

for i in content:
    if 'Train Accuracy' not in i:
        del i

y = [i for i in content if 'Train Accuracy' in i]


#print y
y = map(lambda s: s.strip(), y)
#print y

z = []
for i in y:
    x = i.replace('Train Accuracy','')
    z.append(float(x))

#print z

# t1 = [0.0]*50
# t2 = [0.15]*50 #24
# t3 = [0.25]*50 #42
# t4 = [0.34] #51
# baseline = [0.02]*151
plt.plot(z, label='Training Accuracy')
plt.plot(t1+t2+t3+t4, label='Test Accuracy (taken at intervals of 50 epochs)')
plt.plot(baseline, label='Baseline Accuracy')
plt.legend(prop={'size': 12})
plt.xlabel('Epochs',size=12)
plt.ylabel('Cross-validation Accuracy',size=12)
plt.xticks(np.arange(0,151,10),size=12)
plt.yticks(np.arange(0,1.1,0.1),size=12)
#plt.title('Cross-Validation Accuracy of location predictor',size=14)
plt.show()
