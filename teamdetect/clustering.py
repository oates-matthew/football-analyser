from sklearn.cluster import KMeans


class Clustering:

    def __init__(self):
        self.cluster = KMeans(n_clusters=2, random_state=53)

    def train(self, x):
        if len(x) == 0:
            return []
        return self.cluster.fit_predict(x)

    def assign(self, x):
        if len(x) == 0:
            return []
        return self.cluster.predict(x)
