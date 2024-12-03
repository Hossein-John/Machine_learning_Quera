
import numpy as np

class Preprocessor : 
    def __init__ (self, df):
        self.df = df.copy()
        
    def income_transformer (self) :
        self.df['log_income'] = np.log(self.df['Annual Income (k$)'])
    
    def drop_cols (self) :
        cols_to_drop = ['CustomerID', 'Gender', 'Annual Income (k$)']
        self.df.drop(cols_to_drop, axis=1, inplace=True)        
    
      
    def transform (self) : 
        self.income_transformer()
        self.drop_cols()
        return self.df
