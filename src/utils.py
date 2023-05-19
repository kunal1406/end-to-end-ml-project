import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, X_test, models, epochs, batch_size):
    try:
        report = {}

        for i in range(len(list(models))):

            model = list(models.values())[i]
            model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

            logging.info(f"model summary: {model.summary()}")

            history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.02).history

            logging.info(f"model successfuly trained and below is the object of model history: {history}")

            X_pred = model.predict(X_test)
            X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[2])

            mse = np.mean(np.power(X_test - X_pred, 2), axis=1)

            threshold = np.mean(mse) + 3*(np.std(mse))

            anomalies = np.where(mse > threshold)

            report[list(models.keys())[i]] = max(history['val_accuracy'])

            return report
        
    except Exception as e:
        raise CustomException(e, sys)