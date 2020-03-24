# Uzupełnij kod, aby wybierał losowe cechy z X. Użyj funkcji pick_random_features
X_random_features = pick_random_features(X, 10)
accuracy_2_features["Losowe"] = get_accuracy_list(X_random_features, y)

# Uzupełnij kod, aby wybierał najważniejsze cechy z X. Użyj SelectKBest i fit_transform 
X_reduced_features = SelectKBest(chi2, k=10).fit_transform(X, y)
accuracy_2_features["Wybrane"] = get_accuracy_list(X_reduced_features, y)

# Teraz znajdź najmniej informatywne cechy (piksele) i zobrazuj je na rysunku. 
# Możesz w tym celu użyć funkcji SFS (należy wybrać NAJMNIEJ informatywne cechy)
def plot_mnist(data):
    data = [i for i in data]
    img = np.zeros((28,28), dtype=float)
    for x in range(28):
        for y in range(28):
            img[y][x]=data[y*28+x]
    plt.imshow(img)
    
knn = KNeighborsClassifier(n_neighbors=4)
num_feats = 775
sfsForward = SFS(knn, k_features=num_feats, forward=False, n_jobs=-1)
sfsForward = sfsForward.fit(X, Y)
#print(sfsForward.k_feature_idx_)
features = [1 if i not in sfsForward.k_feature_idx_ else 0 
             for i in range(X.shape[1])]
plot_mnist(features)

# Dokonaj klasyfikacji k-nn na pełnym zbiorze i zbiorze bez m najmniej informatywnych cech. 
# m = 100,200,500
for m in [100, 200, 500]:
    X_train, X_test, y_train, y_test = train_test_split(minst.data, minst.target, 
                                                        train_size=0.7, test_size=0.3, random_state=42)
    sfsForward = SFS(knn, k_features=(X.shape[1]-m), forward=True, n_jobs=-1)
    sfsForward = sfsForward.fit(X_train, t_train)
    X_sfs = [1 if x in sfsForward.k_feature_idx_ else 0 for x in X_train]
    y_sfs = [1 if x in sfsForward.k_feature_idx_ else 0 for x in Y_train]
    print(check_knn_accuracy(X_sfs, X_test, y_sfs, y_test, 4))

# Przetransformować zbiory przy pomocy PCA z N-D do N-D. Jak wyglądają (obrazki) wektory własne odpowiadające największym wartością własnym. 
# Sprawdzić, czy poprawił się wynik klasyfikacji. Dokonać wizualizacji 2-D przy pomocy PCA.
TODO 


# Usunąć m najmniej informatywnych cech PCA. Jak wygląda wynik klasyfikacji.
m = 750
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, 
                            train_size=0.02, test_size=0.1, random_state=42)
X_train_transform, X_test_transform = 
                        pca_transform_dataset(X_train, X_test, X.shape[1]-m)
accuracy = check_accuracy_knn(X_train_transform, X_test_transform, y_train, y_test, 3)
print (accuracy)

# Wybrac m NAJLEPSZYCH cech PCA. Jak wygląda teraz wynik klasyfikacji.
m = 5
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, 
                            train_size=0.02, test_size=0.1, random_state=42)
X_train_t, X_test_t = pca_transform_dataset(X_train, X_test, m)
accuracy = check_accuracy_knn(X_train_t, X_test_t, y_train, y_test, 3)
print (accuracy)

# Wartość m w przypadku wyboru najgorszych cech ma być duże (dla N=784 jakieś m=500), 
# w przypadku wyboru najlepszych małe (m=10-20)
TODO

# Dokonać klasyfikacji z PCA i bez PCA (na pełnym zbiorze cech i zadanym małym M), 
# ale zwiększając ilość przykładów przy pomocy augmentacji (imgaug).
seq = iaa.Sequential([
    iaa.Sometimes(
        0.4, 
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.7*255), per_channel=0.5)),
], random_order=True)
images = [np.array(i, dtype='float32').reshape(28,28) for i in X_train]
noise = seq(images=images)
flat_noise = np.asarray([x.reshape(784) for x in noise])
m = 50
flat_noise = X_train + flat_noise
accuracy = check_accuracy_knn(X_train, X_test, y_train, y_test, 3)
print("no PCA: ", accuracy)
X_train_t, X_test_t = pca_transform_dataset(X_train, X_test, m)
accuracy = check_accuracy_knn(X_train_t, X_test_t, y_train, y_test, 3)
print("PCA: ", accuracy)
