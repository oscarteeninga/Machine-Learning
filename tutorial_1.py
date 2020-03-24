# Przetestuj dokładność klasyfikatora bez operacji na danych treningowych
check_knn_accuracy(X_train, X_test, y_train, y_test)

#Testowany obrazek
def image_mnist(data):
    data = [i for i in data]
    img = np.zeros((28,28), dtype=float)
    for x in range(28):
        for y in range(28):
            img[y][x]=data[y*28+x]
    return img

plt.imshow(image_mnist(X_train[8]))
knn = KNeighborsClassifier(3)
knn.fit(X_train, y_train)
y_predicted = knn.predict(X_test)
print(accuracy_score(y_test, y_predicted))

# Przetestuj dokładność klasyfikatora, stosując binaryzację, np. pixele o odcieniu >=70 oznacz jako 1,
# a pozostałe na 0
#plt.imshow(X_train)
binarized = [[1 if b > 70 else 0 for b in image] for image in X_train]

knn = KNeighborsClassifier(3)
knn.fit(binarized, y_train)
y_predicted = knn.predict(X_test)

plot_mnist(binarized[8])
print(accuracy_score(y_test, y_predicted))

# Przetestuj dokładność klasyfikatora po dodaniu próbek zawierających szum
#plt.imshow(X_train)
noise = iaa.ImpulseNoise(0.15)
noised = noise(images=[X_train])

plot_mnist(noised[0][8])

knn = KNeighborsClassifier(3)
knn.fit(noised[0], y_train)
y_predicted = knn.predict(X_test)
print(y_predicted)
print(accuracy_score(y_test, y_predicted))


# Przetestuj dokładność klasyfikatora po dodaniu próbek poddanych kilku operacjom augmentacji
# (np. dodanie szumu, rozmycia, obrotu, przycięcia)
#plt.imshow(X_train)
def printAccuracy(probe, img, train=y_train):
    knn = KNeighborsClassifier(3)
    knn.fit(img, train)
    y_predicted = knn.predict(X_test)
    print(probe, accuracy_score(y_test, y_predicted))


def array_mnist(img):
    return [img[i][j] for i in range(28) for j in range(28)]
    
#plt.imshow(image_mnist(X_train[8]))

# No effect
printAccuracy("Nothing", X_train, y_train)

# Blur
blur = iaa.GaussianBlur((0.5, 0.7))
test_images = [image_mnist(i) for i in X_train]
probe1 = blur(images=test_images)
#plt.imshow(probe1[8])
test_arrays = [array_mnist(i) for i in probe1]
printAccuracy("Blur", test_arrays)

# Rotation + Blur
rotate = iaa.Affine(rotate=(-10, 10))
test_images = [image_mnist(i) for i in test_arrays]
probe2 = blur(images=test_images)
#plt.imshow(probe2[8])
test_arrays = [array_mnist(i) for i in probe2]
printAccuracy("Rotation + Blur", test_arrays)

# Rotation + Blur + Crop
crop = iaa.Crop(px=(1, 2))
test_images = [image_mnist(i) for i in test_arrays]
probe3 = crop(images=test_images)
#plt.imshow(probe3[8])
test_arrays = [array_mnist(i) for i in probe3]
printAccuracy("Rotation + Blur + Crop", test_arrays)

# Rotation + Blur + Crop + Noise
noise = iaa.ImpulseNoise(0.1)
test_images = [image_mnist(i) for i in test_arrays]
probe4 = noise(images=test_images)
plt.imshow(probe4[8])
test_arrays = [array_mnist(i) for i in probe4]
printAccuracy("Rotation + Blur + Crop + Noise", test_arrays)