# Przetestuj dokładność klasyfikatora bez operacji na danych treningowych
check_knn_accuracy(X_train, X_test, y_train, y_test)

#Testowany obrazek
plt.imshow(X_train)
knn = KNeighborsClassifier(3)
knn.fit(X_train, y_train)
y_predicted = knn.predict(X_test)
print(y_predicted)
print(accuracy_score(y_test, y_predicted))

# Przetestuj dokładność klasyfikatora, stosując binaryzację, np. pixele o odcieniu >=70 oznacz jako 1,
# a pozostałe na 0
#plt.imshow(X_train)
binarized = [[1 if b > 70 else 0 for b in image] for image in X_train]
plt.imshow(binarized)
knn = KNeighborsClassifier(3)
knn.fit(binarized, y_train)
y_predicted = knn.predict(X_test)
print(y_predicted)
print(accuracy_score(y_test, y_predicted))

# Przetestuj dokładność klasyfikatora po dodaniu próbek zawierających szum
#plt.imshow(X_train)
noise = iaa.ImpulseNoise(0.15)
noised = noise(images=[X_train])
plt.imshow(noised[0])
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
    #print(y_predicted)
    print(probe, accuracy_score(y_test, y_predicted))

# No effect
printAccuracy("Nothing", X_train, y_train)
# Blur
probe1 = blur(images=[X_train])
printAccuracy("Blur", probe1[0])
# Rotation + Blur
probe2 = rotate(images=[probe1[0]])
printAccuracy("Rotatnion + Blur", probe2[0])
# Rotation + Blur + Crop
probe3 = crop(images=[probe2[0]])
printAccuracy("Rotation + Blur + Crop", probe3[0])
# Rotation + Blur + Crop + Noise
probe4 = noise(images=[probe3[0]])
printAccuracy("Rotation + Blur + Crop + Noise", probe4[0])