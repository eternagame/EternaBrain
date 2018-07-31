# Test Table 2
Run `predict_acc14.py` to get results for location and `predict_acc14_base.py` to get results for base.

Change the following variable paths to point to the correct directories:
```
path - Change to RNAfold v1.8.5 directory (Line 10)
feats & yloc - Change to test_real_X & y directories (Lines 46-47)
sess & saver - Change to location of CNN model files (Lines 121-123)
```

# Test Table 4
Run `locationCNN.py` and `baseCNN.py` to get training and test accuracies. Cross check this with the results from Table 2.
