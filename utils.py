import pandas as pd
import numpy as np

df = pd.read_csv('./data/results_from_azure.csv')

results = []

i = 1

for x in df.iloc[:, 0].values:
    results.append(str(i) + ',' + str(np.around(x, decimals = 2)))
    i += 1

result = pd.DataFrame(results)

result.to_csv('./data/Video_Games_Results.csv', index=False)

