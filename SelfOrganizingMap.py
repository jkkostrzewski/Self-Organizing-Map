import pandas as pd
from matplotlib import pyplot as plt
import random
import math
from sklearn import preprocessing


# Cell node
class CNode:
    def __init__(self, nr_of_weights, weight_arr=None):
        if weight_arr == None:
            self.weights = [random.random() for i in range(nr_of_weights)]
        else:
            self.weights = [weight_arr[i] for i in range(nr_of_weights)]

    # euclidean distance method
    def get_distance(self, input_array):
        distance = 0
        for i in range(len(self.weights)):
            distance += (input_array[i] - self.weights[i]) ** 2
        return math.sqrt(distance)

    def adjust_weight(self, input_array, learning_rate, influence):
        for i in range(len(self.weights)):
            self.weights[i] += influence * learning_rate * (input_array[i] - self.weights[i])


def plot(data, neurons, color, radius, display_neighbourhood, fig_arr):
    for n in fig_arr:
        n.clear()

    i = 0
    no_of_plots = len(neurons[0].weights)/2
    for fig in fig_arr:
        for j in range(2):
            if no_of_plots > 0:
                ax = fig.add_subplot(2, 1, j+1)
                temp_array_x = []
                temp_array_y = []
                for n in neurons:
                    temp_array_x.append(n.weights[i])
                    temp_array_y.append(n.weights[i+1])
                    if display_neighbourhood:
                        ax.add_artist(plt.Circle((n.weights[0], n.weights[1]), radius, color='b', fill=False))
                ax.plot(data[i+1], data[i+2], 'k.')
                ax.plot(temp_array_x, temp_array_y, '{}o'.format(color))
                i += 2
            no_of_plots -= 1

    plt.pause(0.001)


class SOM:
    def __init__(self, data, no_of_neurons=25, learning_rate=0.5, iterations=1000, initial_neighbourhood=0.5,
                 dead_neur_percent=0.14, delete_neurons=False, display_neighbourhood=False,
                 display_current_iteration=False, display_animation=True, skip_frames_count=1):
        data = data.reset_index()
        x = data.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        self._data = pd.DataFrame(x_scaled)

        self._node_array = [CNode(len(self._data.columns) - 1) for n in range(no_of_neurons)]
        self._winning_node = None
        self._iterations = iterations
        self._current_iteration = 1
        self._influence = 0
        self._start_learning_rate = learning_rate
        self._learning_rate = learning_rate
        self._map_radius = initial_neighbourhood
        self._time_constant = iterations / math.log(initial_neighbourhood)
        self._n_radius = self.calculate_radius()
        self._winning_bool_array = [False for i in range(len(self._node_array))]
        self._error_array = []
        if delete_neurons:
            self._dead_neuron_const = dead_neur_percent * iterations
        else:
            self._dead_neuron_const = 1.01 * iterations
        self._display_neighbourhood = display_neighbourhood
        self._display_current_iteration = display_current_iteration
        self._display_animation = display_animation
        self._skip_frames_count = skip_frames_count

        self._fig_count = int((len(self._data.columns) - 1)/4)
        if (len(self._data.columns) - 1) % 4 == 2:
            self._fig_count += 1

        self._fig_arr = [plt.figure(figsize=(6, 10)) for fig in range(self._fig_count)]

    def find_best_node(self, input_array):
        smallest_distance = self._node_array[0].get_distance(input_array)
        smallest_node = self._node_array[0]
        for node in self._node_array:
            current_dist = node.get_distance(input_array)
            if current_dist < smallest_distance:
                smallest_distance = current_dist
                smallest_node = node

        return smallest_node

    def learning_rate(self):
        return self._start_learning_rate * math.exp(-(self._current_iteration/self._iterations))

    def calculate_radius(self):
        return self._map_radius * math.exp((self._current_iteration / self._time_constant))

    def calculate_error(self):
        distance = 0
        for i in range(len(self._data)):
            point = [self._data.iloc[i, j] for j in range(1, len(self._data.columns))]
            bmu = self.find_best_node(point)
            distance += bmu.get_distance(point)
        self._error_array.append([math.sqrt((distance**2) / len(self._data)), self._current_iteration])

    def reset_bool_array(self):
        self._winning_bool_array = [False for i in range(len(self._node_array))]

    def find_and_set_winning_bool(self, winning_node):
        for i in range(len(self._node_array)):
            if self._node_array[i] == winning_node:
                self._winning_bool_array[i] = True

    def remove_dead_neurons(self):
        end_iterator = len(self._winning_bool_array)
        for i in range(end_iterator):
            if not self._winning_bool_array[i]:
                del self._node_array[i]
                i -= 1
                end_iterator -= 1
                random_int = random.randint(0, len(self._data) - 1)
                random_point = [self._data.iloc[random_int, j] for j in range(1, len(self._data.columns))]
                self._node_array.append(CNode(len(self._data.columns)-1, random_point))
        self.reset_bool_array()

    def run(self):
        while self._current_iteration <= self._iterations:
            if self._display_current_iteration:
                print(self._current_iteration)
            random_int = random.randint(0, len(self._data) - 1)
            random_point = [self._data.iloc[random_int, i] for i in range(1, len(self._data.columns))]

            self._n_radius = self.calculate_radius()

            winning_node = self.find_best_node(random_point)
            self.find_and_set_winning_bool(winning_node)

            if self._current_iteration % self._dead_neuron_const == 0:
                self.remove_dead_neurons()

            width = self._n_radius * self._n_radius

            for node in self._node_array:
                distance_to_node = node.get_distance(winning_node.weights)
                if distance_to_node < width:
                    influence = math.exp(-(distance_to_node ** 2) / (2 * width))
                    node.adjust_weight(random_point, self.learning_rate(), influence)

            self._learning_rate = self.learning_rate()

            self.calculate_error()
            if self._display_animation:
                if self._current_iteration % self._skip_frames_count == 0:
                    plot(self._data, self._node_array, 'b', self._n_radius, self._display_neighbourhood, self._fig_arr)
            self._current_iteration += 1
        if self._display_animation == False:
            plot(self._data, self._node_array, 'b', self._n_radius, self._display_neighbourhood, self._fig_arr)
        plt.show()
        plt.title("Quantization error chart for each epoch")
        plt.plot([row[1] for row in self._error_array], [row[0] for row in self._error_array])
        plt.show()
