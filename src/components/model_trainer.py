import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
import tensorflow as tf
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def autoencoder_model(self, X):
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        L1 = LSTM(16, activation='linear', return_sequences=True, 
                kernel_regularizer=regularizers.l2(0.00))(inputs)
        L2 = Dropout(0.2)(L1)
        L3 = LSTM(4, activation='sigmoid', return_sequences=False)(L1)
        L4 = RepeatVector(X.shape[1])(L3)
        L5 = LSTM(4, activation='sigmoid', return_sequences=True)(L4)
        L6 = Dropout(0.2)(L5)
        L7 = LSTM(16, activation='linear', return_sequences=True)(L5)
        output = TimeDistributed(Dense(X.shape[2]))(L7)    
        model = Model(inputs=inputs, outputs=output)
        return model


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Assigning training and testing array")

            models  = {'LSTM with Autoencoder': self.autoencoder_model(train_array)}

            model_report:dict = evaluate_models(train_array, test_array, models, 4, 32)

            best_model_score = max(sorted(model_report.values()))
            print(best_model_score)
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            best_model.compile(optimizer = 'adam', loss='mse', metrics=['accuracy'])
            history = best_model.fit(train_array, train_array, epochs=4, batch_size=32,
                    validation_split=0.02).history
            return max(history['val_accuracy'])

        except Exception as e:
            raise CustomException(e, sys)


