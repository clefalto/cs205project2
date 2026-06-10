import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)

from ucimlrepo import fetch_ucirepo 

  
# fetch dataset 
zoo = fetch_ucirepo(id=111) 
  
# data (as pandas dataframes) 
X = zoo.data.features 
y = zoo.data.targets 
animal_names = zoo.data.ids


# remove the accursed frog
# get index of first match of frog
dupe_index = (animal_names['animal_name'] == 'frog').idxmax()

animal_names.drop(dupe_index, inplace=True)
X.drop(dupe_index, inplace=True)


legs_oh = pd.get_dummies(X['legs']).astype(int)

legs_oh = legs_oh.add_prefix("legs_")
# print(legs_oh)


X = X.drop(columns=['legs'])
X = pd.concat([X, legs_oh], axis=1)


# print(X)
# print(animal_names)

Z = linkage(pdist(X, 'jaccard'), method='complete')

plt.figure(figsize=(25,10))
plt.title("hierarchical clustering dendrogram")

dendrogram(Z, labels=animal_names['animal_name'].tolist(), orientation='right')
plt.savefig("output_dendro_hires.png", dpi=300, bbox_inches='tight')
plt.show()