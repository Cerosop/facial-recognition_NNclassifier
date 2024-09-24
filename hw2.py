import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def load_dataset(data_path):
    X = []
    y = []
    for i, folder in enumerate(os.listdir(data_path)):
        if 'Non' in folder:
            continue
        for file in os.listdir(os.path.join(data_path, folder)):
            if '.db' in file:
                continue
            img_path = os.path.join(data_path, folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            X.append(img.flatten())
            y.append(i)
    return np.array(X), np.array(y)

data_path = "ORL3232"
X, y = load_dataset(data_path)

X_train = np.array([x for i, x in enumerate(X) if i % 2 == 0])
X_test = np.array([x for i, x in enumerate(X) if i % 2 == 1])
y_train = np.array([x for i, x in enumerate(y) if i % 2 == 0])
y_test = np.array([x for i, x in enumerate(y) if i % 2 == 1])


def preprocess_data(X_train, X_test, n_components=100):
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    lda = LinearDiscriminantAnalysis(n_components=40 - 1)
    X_train_lda = lda.fit_transform(X_train_pca, y_train)
    X_test_lda = lda.transform(X_test_pca)
    
    return X_train_lda, X_test_lda

X_train_processed, X_test_processed = preprocess_data(X_train, X_test)


def nearest_neighbor_classifier(train_data, train_labels, test_data):
    predictions = []
    for test_point in test_data:
        min_distance = float('inf')
        nearest_label = None
        for train_point, label in zip(train_data, train_labels):
            distance = np.linalg.norm(test_point - train_point)
            if distance < min_distance:
                min_distance = distance
                nearest_label = label
        predictions.append(nearest_label)
    return np.array(predictions)

predictions = nearest_neighbor_classifier(X_train_processed, y_train, X_test_processed)

accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)