import csv
import random

from pathlib import Path
from math import sqrt

import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).parent/ 'data' / 'water-treatment.preprocessed'

PRINT_ITERATION = True
PRINT_CLASSIFICATION = True
COLORS = ['cyan', 'pink', 'lime', 'magenta', 'orange', 'yellow', 'red', 'blue', 'brown', 'purple', 'green']


def main():
    cluster_num = None
    while cluster_num is None or cluster_num>len(COLORS) or cluster_num<1:
        try:
            cluster_num = int(input(f'Number of cluster (max. {len(COLORS)}): '))
        except ValueError:
            print('The number you typed is invalid')


    # Loading data
    with open(DATA_PATH, 'r', newline='') as file:
        reader = csv.reader(file)
        cols = tuple(next(reader))
        used_idx = {col_name:cols.index(col_name) for col_name in cols}

        # Casting dataset values to float
        dataset = []
        for row in reader:
            dataset.append(list(map(float, row)))


    # Initializing attribute pairs (plant input and output)
    attribute_pairs = tuple(zip(cols[::2], cols[1::2]))

    # Initializing k-means procedure for each pair
    k_means_list = []
    for x, y in attribute_pairs:
        col_x = tuple(zip(*dataset))[used_idx[x]]
        col_y = tuple(zip(*dataset))[used_idx[y]]
        datapoints = list(Datapoint(coordinate, None) for coordinate in zip(col_x, col_y))

        unique_coordinates = set(datapoint.coordinate for datapoint in datapoints)

        centroids = list(Centroid(coordinate, id) for id, coordinate in enumerate(random.sample(unique_coordinates, cluster_num)))
        k_means_list.append(K_Means(datapoints, centroids))


    # Iterating over K-Means list
    for k_means, pair in zip(k_means_list, attribute_pairs):
        print(f"{f' K-MEANS {pair} ':=^180}")
        stable = False

        while not stable:
            stable = k_means.iterate()

        # Printing centroid movements
        pad = 25
        if PRINT_ITERATION:
            for num, iterations in enumerate(k_means.centroids_at_iter):
                print(f'Iteration {num}')
                print(f"{'cluster':^{pad}}", end='')
                headers = list(f'axis {x}' for x in range(1, len(k_means.dataset[0].coordinate)+1))
                for header in headers:
                    print(f'{header:^{pad}}', end='')
                print()

                for centroid in iterations:
                    print(f'{centroid.cluster_id:^{pad}}', end='')
                    for axis in centroid.coordinate:
                        print(f'{axis:^{pad}}', end='')
                    print()
                print()
            print()
            print()

        # Printing Clustering result
        if PRINT_CLASSIFICATION:
            print('Classifications: ')
            headers = list(f'axis {x}' for x in range(1, len(k_means.dataset[0].coordinate)+1))
            for header in headers:
                print(f'{header:^{pad}}', end='')
            print(f"{'cluster':^{pad}}", end='')
            print()

            for datapoint in k_means.dataset:
                for axis in datapoint.coordinate:
                    print(f'{axis:^{pad}}', end='')
                print(f"{datapoint.classification:^{pad}}", end='')
                print()
            print()
            print()

    # Initialize figure
    fig, axs = plt.subplots(2,2)
    fig.suptitle(f'K-Means with {cluster_num} cluster(s)')

    # Create groups based on clustering
    subplots = [subplot for row in axs for subplot in row]
    for k_means, pair, subplot in zip(k_means_list, attribute_pairs, subplots):
        x, y = pair
        subplot.set_xlabel(x)
        subplot.set_ylabel(y)
        cluster_groups = {centroid.cluster_id:[] for centroid in k_means.centroids_at_iter[-1]}
        for datapoint in k_means.dataset:
            cluster_groups[datapoint.classification].append(datapoint.coordinate)
        
        for (cluster, coordinates), color in zip(cluster_groups.items(), COLORS):
            x_coords, y_coords = tuple(zip(*coordinates)) # Transpose
            subplot.scatter(x_coords, y_coords, color=color, alpha=0.2, label=cluster)

        subplot.legend()
        
    plt.show()

class Datapoint:
    def __init__(self, coordinate, classification):
        self.coordinate = coordinate
        self.classification = classification

    def __iter__(self):
        yield (self.coordinate, self.classification)

class Centroid:
    def __init__(self, coordinate, cluster_id):
        self.coordinate = coordinate
        self.cluster_id = cluster_id

class K_Means:
    def __init__(self, dataset, centroids):
        self.centroids_at_iter = []
        self.centroids_at_iter.append(centroids)
        self.dataset = dataset
        self.iteration = 0


    def iterate(self):
        # Initialize list of members for each centroid
        centroids_members = {centroid.cluster_id:[] for centroid in self.centroids_at_iter[0]}

        # Classifying datapoints
        for datapoint in self.dataset:
            min_distance = None
            for centroid in self.centroids_at_iter[-1]:
                centroid_distance = self._distance(datapoint.coordinate, centroid.coordinate)
                try:
                    if centroid_distance < min_distance:
                        min_distance = centroid_distance
                        centroids_members[datapoint.classification].remove(datapoint)
                        datapoint.classification = centroid.cluster_id
                        centroids_members[centroid.cluster_id].append(datapoint)
                        
                except TypeError:
                    min_distance = centroid_distance
                    datapoint.classification = centroid.cluster_id
                    centroids_members[centroid.cluster_id].append(datapoint)

        # Initializing list of new centroids
        new_centroids = []
        
        # Calculating the coordinate of new centroids
        for cluster_id, datapoints in centroids_members.items():
            coordinate_sum = [0]*len(datapoints[0].coordinate)
            for datapoint in datapoints:
                # Summing each coordinate axis
                for i in range(len(datapoint.coordinate)):
                    coordinate_sum[i]+=datapoint.coordinate[i]

            # Calculate new cordinate by averaging the sum
            new_coordinate = tuple(x/len(datapoints) for x in coordinate_sum)

            # Instantiate new centroid and append it to new centroid list
            new_centroid = Centroid(new_coordinate, cluster_id)
            new_centroids.append(new_centroid)

        stable = True
        for old_centroid, new_centroid in zip(self.centroids_at_iter[-1], new_centroids):
            if old_centroid.coordinate != new_centroid.coordinate:
                stable=False
                break

        # Append new centroids list to centroids at iter
        self.centroids_at_iter.append(new_centroids)

        return stable

    def _distance(self, a, b):
        return sqrt(sum((x2-x1)**2 for x1, x2 in zip(a,b)))



if __name__=='__main__':
    main()