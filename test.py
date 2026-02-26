import pickle
# Check test split
with open('splits/KSDD2/split_53.pyb', 'rb') as f:
    train_samples, test_samples = pickle.load(f)
    
print(f"Defect samples: {sum(1 for p,s in test_samples if s==1)}")
print(f"Non-defect samples: {sum(1 for p,s in test_samples if s==0)}")