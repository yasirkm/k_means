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
    # Asking user for the number of cluster
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
    attribute_pairs = tuple(zip(cols[::2], cols[1::2]))     # Even index is input. Odd index is output

    # Initializing k-means procedure for each pair
    k_means_list = []
    for x, y in attribute_pairs:
        # Getting lists of attribute values
        col_x = tuple(zip(*dataset))[used_idx[x]]   # Transpose dataset list and access its column idx as row
        col_y = tuple(zip(*dataset))[used_idx[y]]

        # Create list of datapoints with the combination of attributes value
        datapoints = list(Datapoint(coordinate, None) for coordinate in zip(col_x, col_y))

        # Create unique coordinates for centroids initial position
        unique_coordinates = set(datapoint.coordinate for datapoint in datapoints)

        # Generate list of cluster_num centroids by sampling unique_coordinates randomly
        centroids = list(Centroid(coordinate, id) for id, coordinate in enumerate(random.sample(unique_coordinates, cluster_num)))
        k_means_list.append(K_Means(datapoints, centroids))


    # Iterating over K-Means list
    for k_means, pair in zip(k_means_list, attribute_pairs):
        print(f"{f' K-MEANS {pair} ':=^180}")   # As separator in console output
        pad = 25    # Pad initialization for console output

        # Iterate until stable (not a single centroid updates its coordinate)
        stable = False
        while not stable:
            stable = k_means.iterate()

        # Printing centroid movements
        if PRINT_ITERATION:
            for iteration_num, centroids in enumerate(k_means.centroids_at_iter):    # Iterate over all recorded iteration and centroids coordinate at said iteration
                print(f'Iteration {iteration_num}')

                # Printing headers
                headers = ['cluster']+list(f'axis {x}' for x in range(1, len(k_means.dataset[0].coordinate)+1))
                for header in headers:
                    print(f'{header:^{pad}}', end='')
                print()

                # Printing values
                for centroid in centroids:
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

            # Printing header
            headers = list(f'axis {x}' for x in range(1, len(k_means.dataset[0].coordinate)+1)) + ['cluster']
            for header in headers:
                print(f'{header:^{pad}}', end='')
            print()

            # Printing values
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
        # Setting axis label
        x, y = pair
        subplot.set_xlabel(x)
        subplot.set_ylabel(y)

        # Iniliatize cluster groups
        cluster_groups = {centroid.cluster_id:[] for centroid in k_means.centroids_at_iter[-1]}

        # Adding datapoints coordinate to their group
        for datapoint in k_means.dataset:
            cluster_groups[datapoint.classification].append(datapoint.coordinate)
        
        # Adding clusters to figure
        for (cluster, coordinates), color in zip(cluster_groups.items(), COLORS):
            # Separate x and y coords
            x_coords, y_coords = tuple(zip(*coordinates)) # Transpose

            # Chart coords
            subplot.scatter(x_coords, y_coords, color=color, alpha=0.2, label=cluster)
        
        subplot.legend(loc='upper right')    # Show subplot labels
        
    plt.show()

class Datapoint:
    '''
        Class for datapoint object.
        Have coordinate and classification attribute.
    '''
    def __init__(self, coordinate, classification):
        '''
            Instantiates Datapoint object.
            coordinate: iterable array-like object.
            classification: alphanumeric or None
        '''
        self.coordinate = coordinate
        self.classification = classification

    def __iter__(self):
        yield (self.coordinate, self.classification)

class Centroid:
    '''
        Class for centroid object.
        Have coordinate and cluster_id attribute.
    '''
    def __init__(self, coordinate, cluster_id):
        '''
            Instantiates Centroid object.
            coordinate: iterable array-like object.
            cluster_id: alphanumeric
        '''
        self.coordinate = coordinate
        self.cluster_id = cluster_id

class K_Means:
    '''
        Class for K-Means process object.
        K_Means objects track their centroids coordinate for each iteration.
    '''
    def __init__(self, dataset, centroids):
        '''
            Instansiate K_Means process object.
            dataset: array-like object for Datapoint class instance(s)
            centroids: list of Datapoint class instance(s)
        '''
        self.dataset = dataset

        # Initializing centroids (iteration 0)
        self.centroids_at_iter = []
        self.centroids_at_iter.append(centroids)
        self.iteration = 0


    def iterate(self):
        # Initialize list of members for each centroid
        centroids_members = {centroid.cluster_id:[] for centroid in self.centroids_at_iter[0]}

        # Classifying datapoints
        for datapoint in self.dataset:
            min_distance = None

            # Iterating over each centroid and classifying datapoint with the closest centroid
            for centroid in self.centroids_at_iter[-1]:
                centroid_distance = self._distance(datapoint.coordinate, centroid.coordinate)
                try:
                    if centroid_distance < min_distance:    # Raise TypeError if min_distance is None
                        min_distance = centroid_distance
                        centroids_members[datapoint.classification].remove(datapoint)
                        datapoint.classification = centroid.cluster_id
                        centroids_members[centroid.cluster_id].append(datapoint)
                        
                except TypeError:   # Indicate the first centroid to be considered
                    min_distance = centroid_distance
                    datapoint.classification = centroid.cluster_id
                    centroids_members[centroid.cluster_id].append(datapoint)

        # Initializing list of new centroids
        new_centroids = []
        
        # Calculating the new coordinate of each centroids
        for cluster_id, datapoints in centroids_members.items():
            # Instantiate list of 0s for the sum of each axis in coordinate
            coordinate_sum = [0]*len(datapoints[0].coordinate)

            # Summing all datapoint coordinate
            for datapoint in datapoints:
                # Summing each coordinate axis
                for i in range(len(datapoint.coordinate)):
                    coordinate_sum[i]+=datapoint.coordinate[i]

            # Calculate new cordinate by averaging the sum
            new_coordinate = tuple(x/len(datapoints) for x in coordinate_sum)

            # Instantiate new centroid and append it to new centroid list
            new_centroid = Centroid(new_coordinate, cluster_id)
            new_centroids.append(new_centroid)

        # Determining the stability of k-means centroids
        stable = True
        for old_centroid, new_centroid in zip(self.centroids_at_iter[-1], new_centroids):
            if old_centroid.coordinate != new_centroid.coordinate:  # If any coordinate of old_centroid and new_centroid is different, then the k-means centroid is not stable yet
                stable=False
                break

        # Append new centroids list to centroids at iter
        self.centroids_at_iter.append(new_centroids)

        return stable

    @staticmethod
    def _distance(a, b):
        '''
            Helper function to evaluate the euclidian distance of coordinate a and b.
            coordinate a and b can have as many dimension as python can handle.
        '''
        return sqrt(sum((x2-x1)**2 for x1, x2 in zip(a,b)))



if __name__=='__main__':
    main()