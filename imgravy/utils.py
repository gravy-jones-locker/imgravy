import cv2
import numpy as np
import math

class Utils:

    """
    A class for not strictly image-related tasks frequently used in the 
    processing pipeline.
    """

    @staticmethod
    def rotate_pt(pt, c, theta):
        """
        Apply the specified rotation to the point passed.
        """
        M = cv2.getRotationMatrix2D(c, theta, 1)

        # The rotation matrix is applied to the point as a vector 
        return np.dot(M, [*pt, 1]).astype('int32')

    @staticmethod
    def cvt_lines2info(lines):
        """
        Return a list of (theta, mag) tuples from a list of lines.
        """
        theta_mags = []  # Create list of line angles and magnitudes
        for (x0, y0), (x1, y1) in lines:
            dx = x1 - x0
            dy = y0 - y1  # Inverted coordinate system
            theta = math.degrees(math.atan(dy / dx))
            if theta < 0:
                theta += 90
            theta_mags.append((theta, np.sqrt(dx ** 2 + dy ** 2)))
        return theta_mags