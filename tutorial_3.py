data_size = 400

mnist_zipped = list(zip(mnist.data, mnist.target))
mnist_random_sample = random.sample(mnist_zipped, data_size)

fmnist_zipped = list(zip(fmnist.data, fmnist.target))
fmnist_random_sample = random.sample(fmnist_zipped, data_size)

tng_zipped = list(zip(tng.data, tng.target))
tng_random_sample = random.sample(tng_zipped, data_size)

data = [mnist_random_sample, fmnist_random_sample, tng_random_sample]
data_names = ["MNIST", "FMINST", "TNG"]

def image_mnist(data):
    data = [i for i in data]
    img = np.zeros((28,28), dtype=np.uint8)
    for x in range(28):
        for y in range(28):
            img[y][x]=data[y*28+x]
    return img

def array_mnist(img):
    return [img[i][j] for i in range(28) for j in range(28)]

def classifier_test(x_train, y_train, x_test, y_test, classifier, classifier_name, data_name, stats):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    stats[data_name][classifier_name] = {}
    stats[data_name][classifier_name]['accuracy'] = accuracy_score(y_test, y_pred)
    stats[data_name][classifier_name]['cross'] = cross_val_score(clf, X, y, cv=5)
    stats[data_name][classifier_name]['precision'] = precision_score(y_test, y_pred, average='macro')
    stats[data_name][classifier_name]['recall'] = recall_score(y_test, y_pred, average='macro')
    stats[data_name][classifier_name]['f1'] = f1_score(y_test, y_pred, average='macro')
    
def data_test(X, Y, data_name):
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    stats[data_name] = {}
    for j in ['sigmoid', 'linear', 'poly', 'rbf']:
        classifier_test(x_train, y_train, x_test, y_test, SVC(kernel=j, C=C), j, data_name, stats)
    classifier_test(x_train, y_train, x_test, y_test, knn, 'knn', data_name, stats)
    
    # augumentacja
    seq = iaa.Sequential([
        iaa.Sometimes(1.0,iaa.GaussianBlur(sigma=(0.5, 0.7))),
        iaa.Affine(rotate=(-10, 10)),
    ])
    
    if data_name == 'MNIST':
        test_images = [image_mnist(i) for i in x_train]
        x_train = [i.reshape(784) for i in seq(images=test_images)]
        
        stats_aug[data_name] = {}
        for j in ['sigmoid', 'linear', 'poly', 'rbf']:
            classifier_test(x_train, y_train, x_test, y_test, SVC(kernel=j, C=C), j, data_name, stats_aug)
        classifier_test(x_train, y_train, x_test, y_test, knn, 'knn', data_name, stats_aug)
    

    
def data_test_tng(X, Y):
    stemmer = WordNetLemmatizer()
    documents = []
    for sen in range(0, len(X)):
        document = re.sub(r'\W', ' ', str(X[sen]))
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        document = re.sub(r'^b\s+', '', document)
        document = document.lower()
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        documents.append(document)
    
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(documents).toarray()
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    data_test(X, Y, 'TNG')
    
def show_result_plot(data_stat):
    for i in range(len(data_stat)):
        for j in ['accuracy', 'precision', 'recall', 'f1']:
            xs = []
            ys = []
            for c in ['sigmoid', 'linear', 'poly', 'rbf', 'knn']:
                xs.append(data_stat[data_names[i]][c][j])
                ys.append(c)
            plt.plot(ys, xs, label = j)
            plt.title(data_names[i])
            plt.xlabel('classificator') 
        plt.legend()
        plt.show()

C = 10
knn = KNeighborsClassifier(n_neighbors=4)

stats = {}
stats_aug = {}

for i in range(3):
    X, y = zip(*(data[i]))
    if (i == 2):
        data_test_tng(X, y)
    else: 
        data_test(X, y, data_names[i])