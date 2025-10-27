from pathlib import Path
import pandas as pd

root = '/home/workspace/yoavellinson/binaural_TSE_Gen/pts'
pts =  list(Path(root).rglob("*.pt"))

df_full = pd.DataFrame(pts,columns=['path'])
train_per = 0.9
train_len = int(train_per*len(df_full))


df_shuffled = df_full.sample(frac=1, random_state=42).reset_index(drop=True)
df_train = df_shuffled.iloc[:train_len].reset_index(drop=True)
df_test = df_shuffled.iloc[train_len:].reset_index(drop=True)

df_train.to_csv(root+'/train.csv')
df_test.to_csv(root+'/test.csv')

print(len(df_full))



