from burst_detection import fwhm_burst_norm
import numpy as np
import unittest


class FWHM_testing(unittest.TestCase):
    def setUp(self):
        self.none = np.nan
        self.peak_value = 10
        self.peak_loc = (50,50)
        self.edge1_loc = (50,99)
        self.corner_loc = (99,99)
        self.square_loc = (40,60)
        self.empty = np.zeros((100,100))
        self.single = np.zeros((100,100))
        self.single[self.peak_loc] = self.peak_value
        self.square = np.zeros((100,100))
        self.square[
            self.square_loc[0]:self.square_loc[1], 
            self.square_loc[0]:self.square_loc[1]
        ] = self.peak_value
        self.edge1 = np.zeros((100,100))
        self.edge1[self.edge1_loc] = self.peak_value
        self.corner = np.zeros((100,100))
        self.corner[self.corner_loc] = self.peak_value
        self.strip = np.zeros((100,100))
        self.strip[50,:] = 10


    def test_empty(self):
        self.assertEqual(
            fwhm_burst_norm(self.empty, (self.peak_loc[0], self.peak_loc[1])),
            (0, 0, 0, 0)
        )

    def test_single_peak(self):
        self.assertEqual(
            fwhm_burst_norm(self.single, (self.peak_loc[0], self.peak_loc[1])),
            (1, 1, 1, 1)
        )
    
    def test_square_peak(self):
        self.assertEqual(
            fwhm_burst_norm(self.square, (self.peak_loc[0], self.peak_loc[1])),
            (10, 10, 10, 10)
        )
    
    def test_edge_peak(self):
        self.assertEqual(
            fwhm_burst_norm(self.edge1, (self.edge1_loc[0], self.edge1_loc[1])),
            (1, 1, 1, 1)
        )
    
    def test_corner_peak(self):
        self.assertEqual(
            fwhm_burst_norm(self.corner, (self.corner_loc[0], self.corner_loc[1])),
           (1, 1, 1, 1)
        )
    
    def test_strip_peak(self):
        self.assertEqual(
            fwhm_burst_norm(self.strip, (self.peak_loc[0], self.peak_loc[1])),
           (self.none, self.none, 1, 1)
        )

if __name__ == "__main__":
    unittest.main()
