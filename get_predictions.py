import pandas as pd

df = pd.read_csv("./workspace/session_20260329_150927/evaluator/predictions.csv")
df.index.name = 'index'
df.to_csv("submition.csv")