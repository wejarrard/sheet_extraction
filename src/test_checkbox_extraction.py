"""test_checkbox_extraction.py

Checks to make sure the output of the checkbox_extraction file remains
the same (ie: logic wasn't accidentally changed when syntax was).

This file requires a csv with the output of the old file called
'Predicted_Values_oldfile.csv' be in the same directory as this file
alongside the output of the changed file.
"""

import unittest
import pandas as pd

from checkbox_extractor import checkbox_extractor


class test_checkbox_extraction(unittest.TestCase):
    def setUp(self):
        test_extractor = checkbox_extractor("../images/flowsheets")
        self.old_file_output = pd.read_csv("Predicted_Values_oldfile.csv")
        self.new_file_output = test_extractor.extract_checkboxes()

    def test_equal_outputs(self):
        """Tests if the outputs of the revised version of the
        checkbox_extraction file are the same.
        """
        for i in self.old_file_output.columns:
            if not self.old_file_output[i].equals(self.new_file_output[i]):
                print(i)
        self.assertTrue(self.old_file_output.equals(self.new_file_output))


if __name__ == "__main__":
    unittest.main()
