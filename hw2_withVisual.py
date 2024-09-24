import os
import cv2
import numpy as np
import plotly.graph_objects as go
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

def preprocess_data(X_train, X_test, y_train, n_components=50):
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    lda = LinearDiscriminantAnalysis(n_components=20)  
    X_train_lda = lda.fit_transform(X_train_pca, y_train)
    X_test_lda = lda.transform(X_test_pca)
    
    return X_train_lda, X_test_lda, pca, lda

X_train_processed, X_test_processed, pca, lda = preprocess_data(X_train, X_test, y_train)

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


def visualize_data2(X_train, y_train, X_test, y_test, predictions):
    fig = go.Figure()

    # train data
    fig.add_trace(go.Scatter3d(
        x=X_train[:, 0], y=X_train[:, 1], z=X_train[:, 2],
        mode='markers',
        marker=dict(size=6, color=y_train, colorscale='Rainbow', opacity=0.7, cmin=0, cmax=39),
        name='Training data'
    ))

    # test data
    fig.add_trace(go.Scatter3d(
        x=X_test[:, 0], y=X_test[:, 1], z=X_test[:, 2],
        mode='markers',
        marker=dict(size=9, color=y_test, colorscale='Rainbow', opacity=0.7, cmin=0, cmax=39),
        name='Test data'
    ))

    #predict data
    fig.add_trace(go.Scatter3d(
        x=X_test[:, 0], y=X_test[:, 1], z=X_test[:, 2],
        mode='markers',
        marker=dict(size=9, color=predictions, colorscale='Rainbow', opacity=0.7, cmin=0, cmax=39),
        name='Predictions'
    ))

    fig.update_layout(
        title="PCA + LDA + Nearest Neighbor Classifier",
        scene=dict(
            xaxis_title="LDA Component 1",
            yaxis_title="LDA Component 2",
            zaxis_title="LDA Component 3"
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [{"visible": [True, True, False]}],
                    "label": "Show Test data",
                    "method": "restyle"
                },
                {
                    "args": [{"visible": [True, False, True]}],
                    "label": "Show Predictions",
                    "method": "restyle"
                }
            ],
            "direction": "down",
            "showactive": True
        }]
    )

    fig.show()

X_train_processed = lda.fit_transform(X_train_processed, y_train)
X_test_processed = lda.transform(X_test_processed)

visualize_data2(X_train_processed, y_train, X_test_processed, y_test, predictions)
