# Uzupełnij kod, aby wybierał losowe cechy z X. Użyj funkcji pick_random_feature
# Uzupełnij kod, aby wybierał najważniejsze cechy z X. Użyj SelectKBest i fit_transform 
X_random_features = pick_random_features(X, 2)
accuracy_2_features["Losowe"] = get_accuracy_list(X_random_features, y)
X_random_features = pick_random_features(X, 5)
accuracy_5_features["Losowe"] = get_accuracy_list(X_random_features, y)
X_reduced_features = SelectKBest(chi2, k = 2).fit_transform(X, y)
accuracy_2_features["Wybrane"] = get_accuracy_list(X_reduced_features, y)
X_reduced_features = SelectKBest(chi2, k = 5).fit_transform(X, y)
accuracy_5_features["Wybrane"] = get_accuracy_list(X_reduced_features, y)

# Teraz znajdź najmniej informatywne cechy (piksele) i zobrazuj je na rysunku. 
# Możesz w tym celu użyć funkcji SFS (należy wybrać NAJMNIEJ informatywne cechy)
def plot_mnist(data):
    plt.imshow(np.array(data).reshape(28,28))
    
all_feats = X.shape[1] # 784
least_informative_feats = 180
n_feats = all_feats - least_informative_feats
rfe_selector = RFE(estimator=LogisticRegression(),
                   n_features_to_select=n_feats, step=10)
rfe_selector.fit(X, Y)
rfe_support = rfe_selector.get_support()
features = [0 if rfe_support[i] else 1 for i in range(X.shape[1])]
plot_mnist(features)

# Dokonaj klasyfikacji k-nn na pełnym zbiorze i zbiorze bez m najmniej informatywnych cech. 
# m = 100,200,500
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7,    
                                   test_size=0.3, random_state=42)
all_feats = X.shape[1] # 784
xs = [0, 100, 200, 500]
ys =[]
for m in xs:
    rfe_selector = RFE(estimator=LogisticRegression(), 
                       n_features_to_select=(all_feats - m), step=10)
    rfe_selector.fit(X, Y)
    accuracy = check_accuracy_knn(rfe_selector.transform(X_train),  
                                rfe_selector.transform(X_test), y_train, y_test)
    ys.append(accuracy)

# Przetransformować zbiory przy pomocy PCA z N-D do N-D. Jak wyglądają (obrazki) wektory własne odpowiadające największym wartością własnym. 
# Sprawdzić, czy poprawił się wynik klasyfikacji. Dokonać wizualizacji 2-D przy pomocy PCA.
scaler = StandardScaler()
pca = PCA(n_components=n)  
scaler.fit(X)
X2 = scaler.transform(X)
pca.fit(X2)
X3 = pca.transform(X2)
plt.imshow(pca.components_[0].reshape(28,28))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,                 
                                                    random_state=42)
print('Accuracy before PCA: ', check_accuracy_knn(X_train, X_test, 
                                                  y_train, y_test))
scaler = StandardScaler()
pca = PCA(n_components=784)
scaler.fit(X_train)
X_train2 = scaler.transform(X_train)
X_test2 = scaler.transform(X_test)
pca.fit(X_train2)
X_train3 = pca.transform(X_train2)
X_test3 = pca.transform(X_test2)
print('Accuracy after PCA: ', check_accuracy_knn(X_train3, X_test3,  
                                                 y_train, y_test))

scaler = StandardScaler()
pca = PCA(n_components=2)
scaler.fit(X)
X1 = scaler.transform(X)
pca.fit(X1)
X2 = pca.transform(X1)
fig = plt.figure(figsize = (16,16))
ax = fig.add_subplot(1,1,1)
ax.scatter(X2[:,0], X2[:,1], c=[int(y) for y in Y])

# Usunąć m najmniej informatywnych cech PCA. Jak wygląda wynik klasyfikacji.
xs = [700, 750, 775, 780]
ys = []
for m in xs:
    X_train_t, X_test_t = pca_transform_dataset(X_train, X_test, X.shape[1]-m)
    accuracy = check_accuracy_knn(X_train_t, X_test_t, y_train, y_test, 3)
    ys.append(accuracy)

     
# Wybrac m NAJLEPSZYCH cech PCA. Jak wygląda teraz wynik klasyfikacji.
xs = [4, 9, 34, 84]
ys = []

for m in xs:
    X_train_t, X_test_t = pca_transform_dataset(X_train, X_test, m)
    accuracy = check_accuracy_knn(X_train_t, X_test_t, y_train, y_test, 3)
    print (m, accuracy)
    ys.append(accuracy)
    
plt.plot(xs, ys) 
plt.xlabel('m') 
plt.ylabel('accuracy') 
plt.show()


# Wartość m w przypadku wyboru najgorszych cech ma być duże (dla N=784 jakieś m=500), 
# w przypadku wyboru najlepszych małe (m=10-20)
TODO

# Dokonać klasyfikacji z PCA i bez PCA (na pełnym zbiorze cech i zadanym małym M), 
# ale zwiększając ilość przykładów przy pomocy augmentacji (imgaug).
seq = iaa.Sequential([
   iaa.Sometimes(
       0.4,
       iaa.AdditiveGaussianNoise
            (loc=0, scale=(0.0, 0.7*255), per_channel=0.5)),
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
