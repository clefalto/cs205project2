import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation

from ucimlrepo import fetch_ucirepo 

  
# fetch dataset 
zoo = fetch_ucirepo(id=111) 
  
# data (as pandas dataframes) 
X = zoo.data.features 
y = zoo.data.targets 

Z = linkage(pdist(X, 'jaccard'))

plt.figure(figsize=(25,10))
plt.title("hierarchical clustering dendrogram")

print(zoo.data.ids)

dendrogram(Z, labels=zoo.data.ids['animal_name'].tolist())
plt.savefig("output_dendro_hires.png", dpi=300, bbox_inches='tight')
plt.show()