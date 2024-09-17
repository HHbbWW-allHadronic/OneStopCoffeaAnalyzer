import awkward as ak
import numpy as np


def angleToNPiToPi(angle):
     rem = np.remainder(angle, 2.0 * np.pi)
     return ak.where(rem < np.pi, rem, 2.0 * np.pi - rem)

