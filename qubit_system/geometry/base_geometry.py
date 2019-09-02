class BaseGeometry:
    def get_distance(self, i: int, j: int) -> float:
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError
