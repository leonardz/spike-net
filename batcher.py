import numpy as np
import sklearn.utils


class Container:
    def __init__(self, x, y):
        assert len(x) == len(y)
        x, y = np.asarray(x), np.asarray(y)
        self.N = len(x)
        self.x, self.y = sklearn.utils.shuffle(x, y)
        self.offset = 0

    def __reset(self):
        self.x, self.y = sklearn.utils.shuffle(self.x, self.y)
        self.offset = 0

    def next_batch(self, batch_size: int):
        assert 0 < batch_size <= self.N

        if self.offset == self.N:
            self.__reset()

        if self.offset + batch_size > self.N:
            remainder = batch_size - (self.N - self.offset)

            last_x = self.x[self.offset:]
            last_y = self.y[self.offset:]

            self.__reset()

            first_x = self.x[:remainder]
            first_y = self.y[:remainder]

            self.offset = remainder

            x = np.concatenate((last_x, first_x))
            y = np.concatenate((last_y, first_y))

            assert len(x) == len(y) == batch_size

            return x, y
        else:
            x = self.x[self.offset:self.offset + batch_size]
            y = self.y[self.offset:self.offset + batch_size]
            self.offset += batch_size

            return x, y

if __name__ == '__main__':
    a = [[1, 2], [3, 4], [5, 6], [7, 8]]
    b = [1, 3, 5, 7]

    data = Container(a, b)
    print(data.next_batch(2))
    print(data.next_batch(3))
    print(data.next_batch(3))
    print(data.next_batch(4))
    print(data.next_batch(4))
