import numpy as np
import time

def load_data(filename):
    """Load dataset. Returns labels (N,) and features (N, M)."""
    data = np.loadtxt(filename)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    return labels, features

def euclidean_distance(a, b):
    """Compute Euclidean distance between two vectors."""
    return np.sqrt(np.sum((a - b) ** 2))

def leave_one_out_accuracy(labels, features, feature_subset):
    """
    1-NN classifier with leave-one-out cross validation.
    feature_subset: list of column indices (0-based) to use.
    Returns accuracy as a float between 0 and 1.
    """
    if len(feature_subset) == 0:
        return 0

    data = features[:, feature_subset]
    n = len(labels)
    correct = 0

    for i in range(n):
        test_point = data[i]
        test_label = labels[i]

        best_dist = float('inf')
        best_label = None

        for j in range(n):
            if i == j:
                continue
            dist = euclidean_distance(test_point, data[j])
            if dist < best_dist:
                best_dist = dist
                best_label = labels[j]

        if best_label == test_label:
            correct += 1

    return correct / n


def test_all_subsets(labels, features, base_set, pool, remove=False):
    """
    Enumerate all possible subsets of adding/removing the features in pool to/from base_set.
    Return the highest accuracy and the feature that was added to base_set to get it. 
    The remove parameter tells it whether or not to add (remove=False) or remove (remove=True) features.
    """
    best = 0.0
    best_feat = -1
    for i in pool:
        current_subset = base_set.copy()
        if remove:
            current_subset.remove(i)
        else:
            current_subset.add(i)
        
        acc = leave_one_out_accuracy(labels, features, list(current_subset))
        print(f"\tusing feature(s) {str(current_subset)} the accuracy is {acc}")
        if acc > best: 
            best = acc
            best_feat = i
    
    return best, best_feat
    

def forward_selection(labels, features):
    """
    (Greedy) Forward selection feature search. Finds a subset of features that give the best accuracy, 
    starting from an empty subset and adding features one at a time. 
    Returns the best accuracy and a Python set containing the features that give that accuracy.
    """
    # start with 0 features, operator is to add one
    total_features = features.shape[1] # first dimension is the instances, second is the features
    feat_pool = set()
    for i in range(total_features): feat_pool.add(i)
    feat_subset = set()

    overall_best_subset = set()
    overall_best = -1
    
    while len(feat_pool) > 0:
        best, best_feat = test_all_subsets(labels, features, feat_subset, feat_pool, False)
        
        feat_subset.add(best_feat)
        feat_pool.remove(best_feat)

        if best > overall_best:
            overall_best = best
            overall_best_subset = feat_subset.copy()
        else:
            print(f"Accuracy has decreased! ({overall_best} -> {best}) Continuing with the best feature from this depth in case of local minima")
        
        print(f"\nbest accuracy was {best} with feature(s) {feat_subset}\n")

    return overall_best, overall_best_subset

def backward_elimination(labels, features):
    """
    (Greedy) Backward elimination feature search. Finds a subset of features that give the best accuracy, 
    starting from a set with all of them and eliminating features one at a time. 
    Returns the best accuracy and a Python set containing the features that give that accuracy.
    """
    # start with all the features, operator is to remove one
    # mostly the same as forward_selection, except we remove features one at a time, and proceed with the subset that gives us the highest accuracy
    total_features = features.shape[1]
    feat_pool = set()
    for i in range(total_features): feat_pool.add(i)

    overall_best_subset = feat_pool.copy()
    overall_best = -1

    while len(feat_pool) > 1: 
        best, best_feat = test_all_subsets(labels, features, feat_pool, feat_pool, True)

        feat_pool.remove(best_feat)

        if best > overall_best:
            overall_best = best
            overall_best_subset = feat_pool.copy()
        else:
            print(f"Accuracy has decreased! ({overall_best} -> {best}) Continuing with the best feature from this depth in case of local minima")
        
        print(f"\nbest accuracy was {best} with feature(s) {feat_pool} (removed feature {best_feat})\n")

    return overall_best, overall_best_subset

def main():
    print("Welcome to our feature selection algorithm!")
    fname = str(input("\nType the name of the file to test (should be in the same folder): "))
    labels, features = load_data(fname) # hope to josh that the file exists
    print("\nSelect the algorithm you want to run: " \
    "\n\t1) Forward selection" \
    "\n\t2) Backward elimination")

    choice = int(input())

    num_instances = features.shape[0]
    num_feats = features.shape[1]

    print(f"\nThis dataset has {num_feats} features (not including the class attribute), with {num_instances} instances.")

    full_subset = [x for x in range(num_feats)]
    base_acc = leave_one_out_accuracy(labels, features, full_subset)
    print(f"Running nearest neighbor with all {num_feats} features, using leave one out cross validation, we get an accuracy of {base_acc}\n")

    print("Beginning search.\n")

    t_start = time.perf_counter()

    if choice == 1:
        best_acc, best_subset = forward_selection(labels, features)
    if choice == 2:
        best_acc, best_subset = backward_elimination(labels, features)
    
    t_end = time.perf_counter()

    print(f"Finished search in {t_end - t_start:.6f} seconds! Overall best accuracy was {best_acc} with feature subset {best_subset}")




if __name__ == "__main__":
    main()
