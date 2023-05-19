import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # def replace_nan(self,X):

    #     X['AIR_FILTER'].replace(0, pd.NA, inplace=True)
    #     X['AIR_FILTER'].interpolate(method='linear', inplace=True)
    #     X['AIR_FILTER'] = X['AIR_FILTER'].fillna(method='bfill')

    #     X['THROTTL_POS'].replace(0, pd.NA, inplace=True)
    #     X['THROTTL_POS'].interpolate(method='linear', inplace=True)
    #     X['THROTTL_POS'] = X['THROTTL_POS'].fillna(method='bfill')

    #     X['ENGINE_AIR_FILTER_DIFF_PRESS_CV'].replace(0, pd.NA, inplace=True)
    #     X['ENGINE_AIR_FILTER_DIFF_PRESS_CV'].interpolate(method='linear', inplace=True)
    #     X['ENGINE_AIR_FILTER_DIFF_PRESS_CV'] = X['ENGINE_AIR_FILTER_DIFF_PRESS_CV'].fillna(method='bfill')

    #     return X

    def replace_outliers_with_nan(self, X):

        useful_columns = ['ENG_FUEL_RATE', 'RT_EXH_TEMP', 'RT-LT_EXH_TEMP', 'TURBO_OUTLET_PRESSURE']
        X = X.loc[:, useful_columns].copy()
        z_scores = zscore(X)
        threshold = 3
        outliers = np.abs(z_scores) > threshold
        df_nan = X.copy()
        df_nan.values[outliers] = np.nan
        
        return df_nan  
    
    def replace_nan_with_rolling_mean(self, X):

        X = X.copy()
        X[X.isnull()] = X.rolling(window=5, min_periods=1).mean()[X.isnull()]
        return X 

    def get_data_transformer_object(self, X):

        '''
        This function is responsible for data transformation

        '''

        try:

            # X = self.replace_nan(X)
            
            
            # other_columns = ['SS_TimeStamp', 'AIR_FILTER', 'AMB_AIR_TEMP', 'ATMOS_PRES', 'BOOST_PRES', 'ENG_LOAD', 'ENG_SPD', 'LT_EXH_TEMP', 'THROTTL_POS', 'TURBO_INLET_PRESSURE', 'ENGINE_AIR_FILTER_DIFF_PRESS_CV', 'TURBO_OUT_TO_IN_DIFF_CV']

            

            logging.info(f"Getting into the pipeline")

            useful_columns_pipepline = Pipeline([
                # ('Handling missing values', FunctionTransformer(self.replace_nan, validate = False)),
                ('outlier_replacement', FunctionTransformer(self.replace_outliers_with_nan, validate = False)),
                ('nan_replacement', FunctionTransformer(self.replace_nan_with_rolling_mean, validate = False)),
                ('normalization', MinMaxScaler(feature_range=(-1, 1)))

            ])

            return useful_columns_pipepline, X

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path, parse_dates= ['SS_TimeStamp'], index_col='SS_TimeStamp', usecols=[0,1,9,10,13])
            test_df = pd.read_csv(test_path, parse_dates= ['SS_TimeStamp'], index_col='SS_TimeStamp', usecols=[0,1,9,10,13])

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing pipeline")

            preprocessing_obj, train_df = self.get_data_transformer_object(train_df)

            scaled_parameters = preprocessing_obj.fit(train_df)
            X_train = scaled_parameters.transform(train_df)
            X_test = scaled_parameters.transform(test_df)

            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

            logging.info("saving preprocessed object")    
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )

            return (
                X_train,
                X_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)

