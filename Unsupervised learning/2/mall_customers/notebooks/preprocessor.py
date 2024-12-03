
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class Preprocessor : 
    def __init__ (self, df):
        self.df = df.copy()
        self.numeric_columns = ['Annual Income (k$)', 'Spending Score (1-100)']
    def drop_unnecessary_columns(self):
        self.df.drop('CustomerID', axis = 1, inplace = True)
        
    def encode_gender(self):
        encoder = LabelEncoder()
        self.df['Gender'] = encoder.fit_transform(self.df['Gender'])
        self.Gender_encoder = encoder
        
    def handle_outlier(self):
        for col in self.numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)
            self.df[col] = np.where((self.df[col] <= upper_bound) & (self.df[col] >= lower_bound), self.df[col], np.nan)
            
    def filling_outlier(self):
        imputer = SimpleImputer(strategy = 'mean')
        self.df[self.numeric_columns] = imputer.fit_transform(self.df[self.numeric_columns])
        self.imputer = imputer
    
    def scale_features(self):
        scale = StandardScaler()
        self.df[self.numeric_columns] = scale.fit_transform(self.df[self.numeric_columns])
        self.scale = scale
        
    def transform (self) : 
        self.drop_unnecessary_columns()
        self.encode_gender()
        self.handle_outlier()
        self.filling_outlier()
        self.scale_features()
        
        return self.df
