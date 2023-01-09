from Sensor.entity import config_entity, artifact_entity
from Sensor.exception import SensorException
from Sensor.logger import logging
from Sensor.predictor import ModelResolver
from Sensor import utilis
import os, sys

class ModelPusher:
    
    def __init__(self, model_pusher_config:config_entity.ModelPusherConfig, 
                data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                model_trainer_artifact:artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} MODEL PUSHER {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_pusher(self,)->artifact_entity.ModelPusherArtifact:
        try:
            #load object
            logging.info(f"Loading transformer, model and target encoder")
            transformer = utilis.load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model = utilis.load_object(file_path=self.model_trainer_artifact.model_path)
            target_encoder = utilis.load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            #model pusher dir
            logging.info(f"Saving model pusher directory")
            utilis.save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            utilis.save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
            utilis.save_object(file_path=self.model_pusher_config.pusher_target_encoder_path, obj=target_encoder)

            #save model dir
            logging.info(f"Saving model in saved model directory")
            transformer_path = self.model_resolver.get_latest_save_transformer_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            target_encoder_path = self.model_resolver.get_latest_save_target_encoder_path()

            utilis.save_object(file_path=transformer_path, obj=transformer)
            utilis.save_object(file_path=model_path, obj=model)
            utilis.save_object(file_path=target_encoder_path, obj=target_encoder)

            model_pusher_artifact = artifact_entity.ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir, saved_model_dir=self.model_pusher_config.saved_model_dir)
            logging.info(f'Model Pusher Artifact: {model_pusher_artifact}')
            return model_pusher_artifact

        except Exception as e:
            raise SensorException(e, sys)