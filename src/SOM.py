from .Grid import Grid


class SOM:
    def __init__(self, grid: Grid, sigma: float, learning_rate: float, epochs: int):
        self.__grid = grid
        self.__initial_sigma = sigma
        self.__initial_learning_rate = learning_rate
        self.__epochs = epochs

    def fit(self, data):
        for epoch in range(self.__epochs):
            decay_rate = epoch / self.__epochs
            sigma = self.__initial_sigma - (self.__initial_sigma * decay_rate)
            learning_rate = self.__initial_learning_rate - (self.__initial_learning_rate * decay_rate)
            print(f'epochs: {epoch}/{self.__epochs}')
            for data_point in data:
                self.__grid.update(data_point, sigma, learning_rate)
