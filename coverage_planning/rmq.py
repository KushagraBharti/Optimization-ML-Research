# data_structures/rmq.py

class FenwickTree:
    """
    Fenwick (Binary Indexed) Tree supporting point updates
    and prefix-minimum queries in O(log n).
    """

    def __init__(self, n: int):
        self.n = n
        # 1-based indexing
        self.fw = [float('inf')] * (n + 1)

    def update(self, i: int, val: float) -> None:
        """
        Set fw[i] = min(fw[i], val) and propagate.
        """
        while i <= self.n:
            if val < self.fw[i]:
                self.fw[i] = val
            i += i & -i

    def query(self, i: int) -> float:
        """
        Return min over fw[1..i].
        """
        res = float('inf')
        while i > 0:
            if self.fw[i] < res:
                res = self.fw[i]
            i -= i & -i
        return res
