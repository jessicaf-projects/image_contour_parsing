import sys
sys.path.insert(0,'../')

from epochs import create_epoch, create_file_info_df
import unittest
import pandas as pd

class TestStringMethods(unittest.TestCase):

    def test_randomization(self):
        """
        Check if the shuffling/randomization is valid with each new epoch
        by comparing the shuffled data in the new dataframe with the existing
        ones.
        :return:
        """
        epoch = create_epoch('../final_data')
        shuffled_df1 = epoch.get_shuffled_df()

        epoch.new_epoch()
        shuffled_df2 = epoch.get_shuffled_df()

        self.assertFalse(shuffled_df1.equals(shuffled_df2))

    # Additional unit testing to be done:
    # - Check that synthetically generated image containing a
    #   rectangular box image gives the expected pixel area when
    #   matching with a contour

        #images, targets = epoch.get_current_batch()
        #while images is not None:
        #    images, targets = epoch.get_current_batch()

    #
    # def test_isupper(self):
    #     self.assertTrue()
    #     self.assertFalse()
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()