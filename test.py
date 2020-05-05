import matplotlib.pyplot as plt
import numpy as np


class Polygon:
    def __init__(self, n, size=1.0):
        self.n = n
        r = size / 2.0 / np.sin(2.0 * np.pi / self.n / 2.0)
        phi = np.linspace(0, 2.0 * np.pi, self.n, endpoint=False)
        x = r * np.sin(phi)
        y = r * np.cos(phi) - r
        self.corners = np.array([r * np.sin(phi), r * np.cos(phi)])
        self.segments = np.array([((x[i], y[i]), (x[(i + 1) % n], y[(i + 1) % n])) for i in range(self.n)])


class Object:
    def __init__(self):
        self.segments = np.empty((0, 2, 2))

    def add_obj(self, object_b):
        self.segments = np.concatenate([self.segments, object_b.segments], axis=0)

    def add_segment(self, coordinates):
        self.segments = np.concatenate([self.segments, np.array([coordinates, ])], axis=0)

    def plt(self):
        for i in range(self.segments.shape[0]):
            plt.plot(self.segments[i, :, 0], self.segments[i, :, 1], 'k', alpha=0.5)

    def center(self):
        x_mean, y_mean = np.mean(self.segments[:, :, 0]), np.mean(self.segments[:, :, 1])
        self.segments[:, :, 0] = self.segments[:, :, 0] - x_mean
        self.segments[:, :, 1] = self.segments[:, :, 1] - y_mean

    def rotate(self,phi):
        rotation_matrix=np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
        self.segments=np.tensordot(self.segments,rotation_matrix,axes=([2],[0]))


obj = Object()
#obj.add_obj(Polygon(3))
for i in range(10):
    obj.add_segment(Polygon(3, 0.1 * (i+1)).segments[1])
#obj.center()
obj.plt()
for i in range(6):
    obj.rotate(2*np.pi/6.0)
    obj.plt()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in range(3, 30):
#    poly = Polygon(i)
#    plt.plot(poly.x, poly.y, 'k')
# plt.show()
