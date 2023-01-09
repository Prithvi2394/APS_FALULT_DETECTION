from Sensor.exception import SensorException
from Sensor.logger import logging
from Sensor.entity import config_entity, artifact_entity
from Sensor.config import TARGET_COLUMN
from typing import Optional
from Sensor import utilis
import pandas as pd 
import numpy as np 
import os, sys
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import LabelEncoder



class DataTransformation:

    def __init__(self, data_transformation_config:config_entity.DataTransformationConfig,
                        data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} DATA TRANSFORMATION {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)


    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            robust_scaler = RobustScaler()
            pipeline = Pipeline(steps=[
                ('Imputer', simple_imputer),
                ('RobustScaler', robust_scaler)
            ])
            return pipeline
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_data_transformation(self,)-> artifact_entity.DataTransformationArtifact:
        try:
            #Reading training and testing file
            logging.info(f"Reading training and testing file")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            #Selecting input feature for train and test column
            logging.info(f"Selecting input feature from train and test file")
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            #Selecting target feature for train and test dataframe
            logging.info(f"Selecting target feature from train and test dataframe")
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            #Fitting and Transformation on target column
            logging.info(f"Creating object of label_encoder for Target Feature")
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)
                        
            logging.info(f"Transforming train and test dataset using object of label encoder")
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)
            logging.info(f"Completed the fit and transform of target feature")

            #Fiting and Transforming input features
            logging.info(f"Fitting the input train feature")
            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            logging.info(f"Transforming train and test dataset")
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)
            logging.info(f"Completed the fit and transform of input feature")
            
            #Handling the Imbalance dataset using SMOTETomek
            logging.info(f"Handling the imbalamce in the dataset")
            smt = SMOTETomek(random_state=42)
            logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target: {target_feature_train_arr}")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target: {target_feature_train_arr}")

            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target: {target_feature_test_arr}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target: {target_feature_test_arr}")

            #Target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]
            
            #save numpy array
            utilis.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path, array=train_arr)
            utilis.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path, array=test_arr)

            utilis.save_object(file_path=self.data_transformation_config.transform_object_path, obj=transformation_pipeline)
            utilis.save_object(file_path=self.data_transformation_config.target_encoder_path, obj=label_encoder)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path, 
                transformed_train_path=self.data_transformation_config.transformed_train_path, 
                transformed_test_path=self.data_transformation_config.transformed_test_path, 
                target_encoder_path=self.data_transformation_config.target_encoder_path)

            logging.info(f"Data Transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)