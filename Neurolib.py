import matplotlib.pyplot as plt
import numpy as np
import PIL
import random
import math


# for creating models
def rdk():
    return random.random() * 2 - 1


# for showing progress
def progress_bar(percentage):
    symbols = ["  "]
    symbols += ["▏", "▎", "▍", "▋", "▊", "▉"]

    full = percentage // 10
    part = percentage % 10
    six_part = int(part / 10 * 6)
    if full == 10:
        return '[' + symbols[-1] * 10 + ']' + "100%"
    return '[' + symbols[-1] * full + symbols[six_part] + symbols[0] * (9 - full) + "] " + str(percentage) + '% '


# gen each layer in unique file in model directory(path)
def gen_neuro_layers(layers, path="neuro_united"):
    path += '/'
    l_f = open(path + f"l", "w")
    l_f.write("\t".join(map(str, layers)))
    for i in range(len(layers) - 1):
        w_f = open(path + f"w_{i}_{i + 1}", "w")
        b_f = open(path + f"b_{i}_{i + 1}", "w")
        for j in range(layers[i]):
            w_f.write("\t".join(str((rdk())) for _ in range(layers[i + 1])))
            if j == layers[i] - 1:
                continue
            w_f.write("\n")
        b_f.write("\t".join(str(rdk()) for _ in range(layers[i + 1])))
        b_f.close()
        w_f.close()


# class
class NeuroNet:
    # load layers coefficients from model directory
    def __init__(self, path):
        path += '/'
        self.path = path
        self.layers_size = []
        self.weights = []
        self.adds = []
        with open(path + "l", "r") as file:
            self.layers_size = list(map(int, file.readline().split("\t")))
        for i in range(len(self.layers_size) - 1):
            with open(path + f"w_{i}_{i + 1}", "r") as file:
                self.weights.append(np.array(list(map(lambda x: list(map(float, x.split("\t"))), file.read().split("\n")))))
            with open(path + f"b_{i}_{i + 1}", "r") as file:
                self.adds.append(np.array(list(map(lambda x: list(map(float, x.split("\t"))), file.read().split("\n")))))

    # activation function
    def activate_layer(self, layer):
        func = np.vectorize(math.tanh)
        return func(layer)

    def evaluate_layer(self, current_values, number_of_layer):
        res = np.dot(current_values, self.weights[number_of_layer]) + self.adds[number_of_layer]
        return self.activate_layer(res)

    # return all layers value
    def get_results(self, inputs):
        results = list()
        results.append(np.array(inputs))
        for i in range(0, len(self.layers_size) - 1):
            results.append(self.evaluate_layer(results[i], i))
        return results

    # returns only output layer value
    def get_result(self, inputs):
        result = inputs
        for i in range(0, len(self.layers_size) - 1):
            result = self.evaluate_layer(result, i)
        return result

    # train one iteration on one input
    def learn(self, inputs, outputs, speed_of_learning):
        results = self.get_results(inputs)
        curr_err = outputs - results[-1]
        derivative_of_activation_func = np.vectorize(lambda x: 1 - x ** 2)
        for i in range(1, len(self.layers_size)):
            in_layer_err = curr_err * derivative_of_activation_func(results[-i])
            self.adds[-i] += speed_of_learning * in_layer_err
            self.weights[-i] += speed_of_learning * results[-i - 1].T.dot(in_layer_err)
            curr_err = in_layer_err.dot(self.weights[-i].T)

    # train one iteration on some inputs
    def learn_on_range_of_data(self, size_of_data_range, inputss, outputss, speed_of_learning):

        change_of_adds = [np.zeros(np.shape(layer)) for layer in self.adds]
        change_of_weights = [np.zeros(np.shape(layer)) for layer in self.weights]

        for j in range(size_of_data_range):
            results = self.get_results(inputss[j])
            curr_err = outputss[j] - results[-1]
            derivative_of_activation_func = np.vectorize(lambda x: 1 - x ** 2)
            for i in range(1, len(self.layers_size)):
                in_layer_err = curr_err * derivative_of_activation_func(results[-i])
                change_of_adds[-i] += speed_of_learning * in_layer_err / size_of_data_range
                change_of_weights[-i] += speed_of_learning * results[-i - 1].T.dot(in_layer_err) / size_of_data_range
                curr_err = in_layer_err.dot(self.weights[-i].T)
            for i in range(1, len(self.layers_size)):
                self.adds[-i] += change_of_adds[-i]
                self.weights[-i] += change_of_weights[-i]

    def save_neuro(self):
        for i in range(len(self.layers_size) - 1):
            w_f = open(self.path + f"w_{i}_{i + 1}", "w")
            b_f = open(self.path + f"b_{i}_{i + 1}", "w")
            for j in range(self.layers_size[i]):
                w_f.write("\t".join(str(self.weights[i][j, k]) for k in range(self.layers_size[i + 1])))
                if j == self.layers_size[i] - 1:
                    continue
                w_f.write("\n")
            b_f.write("\t".join(str(self.adds[i][0, k]) for k in range(self.layers_size[i + 1])))
            b_f.close()
            w_f.close()