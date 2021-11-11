from numba import jit
import numpy as np
from shapely.geometry import Polygon,MultiPoint

class IOU():
    def __init__(self, object1, object2):
        box1 = np.array(object1[:8], dtype = np.float32).reshape(4,2)
        box2 = np.array(object2[:8], dtype = np.float32).reshape(4,2)
        self.box1 = Polygon(box1).convex_hull
        self.box2 = Polygon(box2).convex_hull
        self.union_box = MultiPoint(np.concatenate((box1, box2))).convex_hull
    
    def __call__(self):
        if not self.box1.intersects(self.box2):
            return 0
        else:
            try:
                inter_area = self.box1.intersection(self.box2).area
                union_area = self.union_box.area

                return float(inter_area)/union_area
            except Exception as e:
                print(e)

# @jit(nopython=True)
def quadrangle_intersection_over_union(A, B):
    return IOU(A, B)()