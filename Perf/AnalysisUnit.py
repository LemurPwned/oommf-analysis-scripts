
import matplotlib as mpl
import pandas as pd

class AnalysisUnit:
    def __init__(self):
        pass

    def save_object(self, object_type, savename):
        if type(object_type) == mpl.figure.Figure:
            object_type.savefig(savename + '.png')
            return True
        elif type(object_type) == pd.DataFrame:
            object_type.to_pickle(savename + '.pkl')
            return True
        return False
