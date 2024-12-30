import classes
import feature_utils
import sys
sys.path.append('./src/')


class sequence_builder:

    def __init__(self):
        self.dataset=None
        
    def fill_dataset(self,dir_name=None):
        self.dataset=classes.Dataset
