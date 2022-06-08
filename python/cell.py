import numpy as np

class Cell:
    def __init__(self):

        self._pose = np.array([[0.0], [0.0], [0.0]])
        self._points= 0
    def set_pose(self, pose):
        self._pose = pose
  