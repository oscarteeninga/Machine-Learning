# Z1
import math
estimators = np.arange(5, 105, 15)
samples = np.arange(1, 10, 1)
features = np.arange(1, 10, 1) 

def ensemble_scores_random_patching(estimators, samples, features):
    results = []
    for n_estimators in estimators:
        for sample in samples:
            for feature in features:
                clf = BaggingClassifier(base_estimator=SVC(),
                            n_estimators=5,
                            bootstrap_features=True,
                            max_features=feature/10,
                            max_samples=sample/10).fit(X_train, y_train)
                result = [n_estimators, sample/10, feature/10, math.floor(1000*sample/feature)/1000, clf.score(X_test, y_test)]
                results.append(result)
    return results

values = ensemble_scores_random_patching(estimators, samples, features)
import pandas as pd
pd.set_option('display.max_rows', 200)
print(pd.DataFrame(data=values, columns=["N", "SAMPLES", "FEATURES", "S/F", "ACC"]).tail(20))

# Z2
standard_scaler = StandardScaler()
X_scaled = standard_scaler.fit_transform(X)
pca = PCA(30)
X_pca = pca.fit_transform(X_scaled)
X_embedded = TSNE(n_components=2).fit_transform(X_pca)
X_embedded.shape
ids = [str(i) for i in range(10)]
plt.figure(figsize=(15, 10))

for c, label in zip(mcolors.TABLEAU_COLORS, ids):
    plt.scatter(X_embedded[y == label, 0], X_embedded[y == label, 1], c=c, label=label)
plt.legend()
plt.show()