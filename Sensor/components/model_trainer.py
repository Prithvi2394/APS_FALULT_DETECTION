from Sensor.entity import config_entity, artifact_entity
from Sensor.exception import SensorException
from Sensor.logger import logging
from Sensor import utilis
from typing import Optional
import os, sys
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

class ModelTrainer:

    def __init__(self, model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        
        try:
            logging.info(f"{'>>'*20} MODEL TRAINER {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def train_model(self, x, y):
        try:
             xgb_clf = XGBClassifier()
             xgb_clf.fit(x,y)
             return xgb_clf
        except Exceptiona as e:
            raise SensorException(e, sys)

    def initiate_model_trainer(self, )->artifact_entity.ModelTrainerArtifact:
        try:
            #Loading trian and test array
            logging.info(f"Loading train and test array")
            train_arr = utilis.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utilis.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            #Splitting input and target feature from both train and test arr
            logging.info(f"Splitting input and target feature from both train and test array")
            x_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            x_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            #Train the model
            logging.info(f'Model Training')
            model = self.train_model(x=x_train, y=y_train)

            #Calculating F1 score
            logging.info(f"Calculating F1 Train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f"Calculatin F1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test)
            logging.info(f"Train score:{f1_train_score} and the Test score: {f1_test_score}")

            #Check for overfitting and underfitting or expected score
            logging.info(f"Check for overfitting and underfitting or expected score")
            logging.info(f"Checking if the model is underfitting or not")
            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give expected accuracy: {self.model_trainer_config.expected_score} and the model actual score is: {f1_test_score}")
            
            logging.info(f"Checking if the model overfitting or not")
            diff = abs(f1_train_score - f1_test_score)

            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")
                
            #Save the trained model
            logging.info(f"Save the trained model")
            utilis.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            #Prepare model trainer artifact
            logging.info(f"Prepare model trainer artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, f1_train_score=f1_train_score, f1_test_score=f1_test_score)
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)