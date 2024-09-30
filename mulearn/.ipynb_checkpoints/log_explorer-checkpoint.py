import pandas as pd
import numpy as np
import json

class LogExplorer:
    
    def __init__(self, filepath):
        
        file = open(filepath, "r+")
        file.seek(0)
        self.file_str = file.read()
        file.close()

        self.file_str = '[' + self.file_str[:-2] + ']'
        

    def to_dict(self):
        return json.loads(self.file_str)


    def make_row(self, log, df, i):

        for elem in log.items():
    
            if isinstance(elem[1], str) or isinstance(elem[1], float) or isinstance(elem[1], int):

                df.loc[i, elem[0]] = None
                df[elem[0]] = df[elem[0]].astype(type(elem[1]))
                df.loc[i, elem[0]] = elem[1]
    
            else:
                self.make_row(elem[1], df, i)

        return 0
    
    def to_pandas(self):
        
        d = self.to_dict()
        df = pd.DataFrame(dtype=object)

        i = 0
        
        for log in d:    
            self.make_row(log, df, i)
            i+=1

        return df       