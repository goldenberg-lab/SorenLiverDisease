import numpy as np
import pandas as pd 
from sklearn.metrics import roc_auc_score

# Step 1: Vary Negative class prevalence from 5% - 95%
# Step 2: Vary Positive class prevalence from 95% - 5%
# Step 3: Assign a score uniformly at random with 5% overlap (Neg class: 0% - 55%, Pos Class: 45% - 199%)
# Step 4. Calculate AUC and plot 

for p_neg in range(5,100,5):
    p_pos = 100 - p_neg
    
    # Generate fake data 
    neg_examples = pd.DataFrame(np.random.uniform(0, 0.65, size=p_neg*10)).rename(columns={0: 'pred'})
    neg_examples['target'] = 0
    pos_examples = pd.DataFrame(np.random.uniform(0.35, 1, size=p_pos*10)).rename(columns={0: 'pred'})
    pos_examples['target'] = 1
    
    data = pd.concat([neg_examples, pos_examples])
    
    auc = roc_auc_score(data['target'], data['pred'])
    msg = f"p_neg: {p_neg}%, p_pos: {p_pos}%, auc: {auc}"
    print(msg)
    
    
