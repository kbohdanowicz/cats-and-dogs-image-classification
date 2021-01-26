import numpy as np
from time import time

from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier


def run_knn(train_test_data: tuple, name: str, find_best: bool):
    (train, test, train_labels, test_labels) = train_test_data

    print('[INFO]: running KNN')
    print(f"[INFO]: image type: '{name}'")

    if find_best:
        model = KNeighborsClassifier(n_jobs=-1)
        params = {'n_neighbors': np.arange(1, 31, 3), 'metric': ['euclidean', 'cityblock']}
        grid = RandomizedSearchCV(model, params)

        start = time()
        grid.fit(train, train_labels)
        accuracy = grid.score(test, test_labels)

        print('[INFO]: randomized search took {:.2f} seconds'.format(time() - start))
        print('[INFO]: grid search accuracy: {:.2f}%'.format(accuracy * 100))
        print('[INFO]: randomized search best parameters: {}'.format(grid.best_params_))

        best_neighbor_count = grid.best_params_['n_neighbors']
        best_metric = 2
        if grid.best_params_['metric'] == 'cityblock':
            best_metric = 1

        best_model = KNeighborsClassifier(n_neighbors=best_neighbor_count, p=best_metric)

        start = time()
        best_model.fit(train, train_labels)
        accuracy = best_model.score(test, test_labels)

        print('[INFO]: best model took {:.2f} seconds to run\n'.format(time() - start))
        print('[INFO]: best model accuracy: {:.2f}%'.format(accuracy * 100))
    else:
        model = KNeighborsClassifier(n_jobs=-1, n_neighbors=1)
        start = time()
        model.fit(train, train_labels)
        accuracy = model.score(test, test_labels)
        print('[INFO]: model took {:.2f} seconds to run'.format(time() - start))
        print('[INFO]: classifier accuracy: {:.2f}%'.format(accuracy * 100))

    print()
