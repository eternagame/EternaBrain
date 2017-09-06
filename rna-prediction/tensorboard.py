from matplotlib import pyplot as plt
import seaborn; seaborn.set()
import os
import numpy as np

with open(os.getcwd()+'/locationCNN14.out') as f:
    content = f.readlines()

for i in content:
    if 'Train Accuracy' not in i:
        del i

y = [i for i in content if 'Train Accuracy' in i]


print y
y = map(lambda s: s.strip(), y)
print y

z = []
for i in y:
    x = i.replace('Train Accuracy','')
    z.append(float(x))

print z

t1 = [0.0]*50
t2 = [0.15]*50
t3 = [0.25]*50
t4 = [0.34]
baseline = [0.02]*151
plt.plot(z, label='Training Accuracy')
plt.plot(t1+t2+t3+t4, label='Test Accuracy (taken at intervals of 50 epochs)')
plt.plot(baseline, label='Baseline Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Cross-validation Accuracy')
plt.xticks(np.arange(0,151,10))
plt.yticks(np.arange(0,1.1,0.1))
plt.title('Cross-Validation Accuracy of location predictor')
plt.show()
