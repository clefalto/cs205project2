import numpy as np


def load_data(filename):
    data = np.loadtxt(filename)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    return labels, features


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def leave_one_out_accuracy(labels, features, feature_subset):
    if len(feature_subset) == 0:
        return 0
    data = features[:, list(feature_subset)]
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


def forward_selection(labels, features, exclude=set()):
    """
    Forward selection starting from an empty set, ignoring any features in exclude.
    All printed indices are original (0-based) feature indices.
    """
    total_features = features.shape[1]
    feat_pool = set(range(total_features)) - exclude
    feat_subset = set()
    overall_best_subset = set()
    overall_best = -1

    while len(feat_pool) > 0:
        best = 0.0
        best_feat = -1
        for i in feat_pool:
            candidate = feat_subset | {i}
            acc = leave_one_out_accuracy(labels, features, candidate)
            print(f"\tusing feature(s) {candidate} the accuracy is {acc}")
            if acc > best:
                best = acc
                best_feat = i

        feat_subset.add(best_feat)
        feat_pool.remove(best_feat)

        if best > overall_best:
            overall_best = best
            overall_best_subset = feat_subset.copy()
        else:
            print(f"Accuracy has decreased! ({overall_best} -> {best}) Continuing in case of local minima")

        print(f"\nbest accuracy was {best} with feature(s) {feat_subset}\n")

    return overall_best, overall_best_subset


def main():
    print("Forward Selection — Ablation Study")
    print("Excludes top-3 features from original run: {27, 13, 21}")
    print("(worst concave points, area SE, worst perimeter)\n")

    fname = input("Type the name of the file to test: ")
    labels, features = load_data(fname)

    num_instances, num_feats = features.shape
    print(f"\nDataset: {num_feats} features, {num_instances} instances.")

    exclude = {27, 13, 21}
    available = set(range(num_feats)) - exclude
    base_acc = leave_one_out_accuracy(labels, features, available)
    print(f"Baseline accuracy using all {len(available)} available features: {base_acc}\n")
    print("Beginning search.\n")

    best_acc, best_subset = forward_selection(labels, features, exclude=exclude)
    print(f"Finished search! Overall best accuracy was {best_acc} with feature subset {best_subset}")


if __name__ == "__main__":
    main()