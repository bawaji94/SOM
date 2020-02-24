from numpy import random, sqrt, exp, square
from sys import maxsize as max_number
import matplotlib.pyplot as plt


def distance(point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2

    delta_x = x1 - x2
    delta_y = y1 - y2

    return sqrt(delta_y ** 2 + delta_x ** 2)


def influence(point1, point2, sigma):
    return exp(-distance(point1, point2) / sigma ** 2)


class Grid:
    def __init__(self, x, y, dimensions):
        self.__x = x
        self.__y = y
        self.__grid = random.rand(x, y, dimensions)

    def update(self, data_point, sigma, learning_rate):
        bmu = self.__find_bmu(data_point)
        for i in range(self.__x):
            for j in range(self.__y):
                current_point = (i, j)
                if distance(current_point, bmu) > sigma:
                    continue

                change = learning_rate * influence(current_point, bmu, sigma) * data_point - self.__grid[i][j]

                self.__grid[i][j] = self.__grid[i][j] + change

    def __find_bmu(self, data_point):
        minimum_distance = max_number
        bmu = (0, 0)
        for i in range(self.__x):
            for j in range(self.__y):
                delta = data_point - self.__grid[i][j]
                distance = sqrt(square(delta).sum())
                if minimum_distance > distance:
                    minimum_distance = distance
                    bmu = (i, j)

        return bmu

    def plot(self):
        plt.figure()
        plt.imshow(self.__grid, origin='bottom')
        plt.title(f'Grid size: {self.__x}x{self.__y}')
        plt.show()
