class BaseGeometry:
    def get_distance(self, i: int, j: int) -> float:
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError
