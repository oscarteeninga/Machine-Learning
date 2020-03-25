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
noise = iaa.ImpulseNoise(0.1)
noised = noise(images=[X_train])

plot_mnist(noised[0][8])

knn = KNeighborsClassifier(3)
knn.fit(noised[0], y_train)
y_predicted = knn.predict(X_test)
print(accuracy_score(y_test, y_predicted))


def array_mnist(img):
    return [img[i][j] for i in range(28) for j in range(28)]

def printAccuracy(probe, img, train=y_train):
    knn = KNeighborsClassifier(3)
    knn.fit(img, train)
    y_predicted = knn.predict(X_test)
    print(probe, accuracy_score(y_test, y_predicted))
    
def printImageAccuracy(probe, img_train, train=y_train):
    img = [array_mnist(i) for i in img_train]
    knn = KNeighborsClassifier(3)
    knn.fit(img, train)
    y_predicted = knn.predict(X_test)
    print(probe, accuracy_score(y_test, y_predicted))
    
#plt.imshow(image_mnist(X_train[8]))

# No effect
test_images = [image_mnist(i) for i in X_train]
printImageAccuracy("Nothing", test_images, y_train)

#plt.imshow(test_images[8])

# Blur
blur = iaa.GaussianBlur((0.5, 0.7))
probe1 = blur(images=test_images)
printImageAccuracy("Blur", probe1)

# Rotation + Blur
rotate = iaa.Affine(rotate=(-10, 10))
probe2 = blur(images=test_images)
#plt.imshow(probe2[8])
printImageAccuracy("Rotation + Blur", probe2)

# Rotation + Blur + Crop
crop = iaa.Crop(px=(1, 2))
test_images = [image_mnist(i) for i in test_arrays]
probe3 = crop(images=test_images)
#plt.imshow(probe3[8])
printImageAccuracy("Rotation + Blur + Crop", probe3)

# Rotation + Blur + Crop + Noise
noise = iaa.ImpulseNoise(0.1)
probe4 = noise(images=test_images)
plt.imshow(probe4[8])
printImageAccuracy("Rotation + Blur + Crop + Noise", probe4)