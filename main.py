import argparse
import os

import tensorflow as tf
from tensorflow import keras


from models import models as models
from utils import losses as losses
from utils import utils as utils

import numpy as np
from IPython import embed



if __name__ == '__main__':


    ap = argparse.ArgumentParser(description="This script runs training of neural networks for EV energy consumption tasks")
    ap.add_argument("-m", "--model", required=True,
        help="model to train")
    ap.add_argument("-d", "--datadir", required=True,
        help="path to dataset")
    ap.add_argument("-b", "--batchsize", required=True,
        help="selected training batch size")
    ap.add_argument("-e", "--epochs", required=True,
        help="number of epochs (MAX if early stopping) for training")
    ap.add_argument("-l", "--lossfunc", required=True,
        help="loss function used for training")
    ap.add_argument("-s", "--speedprofile", required=True,
        help="speed profile type: speed,	speed_limit,	speed_avg_week_time,	speed_avg")
    ap.add_argument("-u", "--summed", required=True,
        help="is data is summed in trip")        
    ap.add_argument("-n", "--name", required=False,
        help="fixed experiment name")        
    ap.add_argument("-o", "--outputdir", required=False,
        help="experiment output folder (for storing models, logs, etc.)")
    ap.add_argument("-i", "--epochsaveinterval", required=False,
                    help="number of epochs between saves")
    ap.add_argument("-p", "--optimizer", required=True,
                    help="Select which optimizer to use (default one is 'adam')")
    args = vars(ap.parse_args())

    ARG_MODEL_NAME = args["model"]
    ARG_DATA_DIR = args["datadir"]
    ARG_BATCH_SIZE = args["batchsize"]
    ARG_EPOCHS = args["epochs"]
    ARG_LOSS_NAME = args["lossfunc"]
    ARG_SPEED_TYPE = args["speedprofile"]
    ARG_SUMMED = args["summed"]    
    ARG_EXPERIMENT_NAME = args["name"]
    ARG_OUTPUT_DIR = args["outputdir"]
    ARG_EPOCH_SAVE_INTERVAL = args["epochsaveinterval"]
    ARG_OPTIMIZER_NAME = args["optimizer"]

 

    output_dim_map = {
        "mse" : 1,
        "ll" : 2
    }
    OUTPUT_DIM = output_dim_map.get(ARG_LOSS_NAME)   

    dataset_map = {
        "al" : 30,
        "pi" : 30
    }
    FEATURE_DIM = dataset_map.get(ARG_DATA_DIR)   


    TIME_STEPS = 550
    PATCH = 10
    SUMMED = int(ARG_SUMMED)

    MODEL_NAME = ARG_MODEL_NAME + "_" + ARG_LOSS_NAME 
    MODEL_NAME_FULL = MODEL_NAME + "_" + str(SUMMED)
    print("----> ", MODEL_NAME_FULL)

    model_map = {
        "dnn_mse_0" : models.dnn(FEATURE_DIM, OUTPUT_DIM, 0, TIME_STEPS),
        "dnn_ll_0" : models.dnn(FEATURE_DIM, OUTPUT_DIM, 0, TIME_STEPS),
        "dnn_mse_1" : models.dnn(FEATURE_DIM, OUTPUT_DIM, 1, TIME_STEPS),
        "dnn_ll_1" : models.dnn(FEATURE_DIM, OUTPUT_DIM, 1, TIME_STEPS),        
        "lstm_mse_0" : models.lstm(TIME_STEPS, FEATURE_DIM, 0, OUTPUT_DIM),
        "lstm_ll_0" : models.lstm(TIME_STEPS, FEATURE_DIM, 0, OUTPUT_DIM), 
        "lstm_mse_1" : models.lstm(TIME_STEPS, FEATURE_DIM, 1, OUTPUT_DIM),
        "lstm_ll_1" : models.lstm(TIME_STEPS, FEATURE_DIM, 1, OUTPUT_DIM),         
        "transformer_mse_0" : models.create_sig_classifier(TIME_STEPS, PATCH, int(ARG_BATCH_SIZE), 0, OUTPUT_DIM),
        "transformer_ll_0" : models.create_sig_classifier(TIME_STEPS, PATCH, int(ARG_BATCH_SIZE), 0, OUTPUT_DIM),
        "transformer_mse_1" : models.create_sig_classifier(TIME_STEPS, PATCH, int(ARG_BATCH_SIZE), 1, OUTPUT_DIM),
        "transformer_ll_1" : models.create_sig_classifier(TIME_STEPS, PATCH, int(ARG_BATCH_SIZE), 1, OUTPUT_DIM),        
    }
    model = model_map.get(MODEL_NAME_FULL)

    loss_map = {
        "dnn_mse_0" : tf.keras.losses.MeanSquaredError(),
        "dnn_ll_0" : losses.gaussian_nll,
        "dnn_mse_1" : tf.keras.losses.MeanSquaredError(),
        "dnn_ll_1" : losses.gaussian_nll,        
        "lstm_mse_0" : tf.keras.losses.MeanSquaredError(),   
        "lstm_ll_0" : losses.gaussian_nll_lstm,
        "lstm_mse_1" : tf.keras.losses.MeanSquaredError(),   
        "lstm_ll_1" : losses.gaussian_nll,        
        "transformer_mse_0" : tf.keras.losses.MeanSquaredError(),      
        "transformer_ll_0" : losses.gaussian_nll_lstm,
        "transformer_mse_1" : tf.keras.losses.MeanSquaredError(),      
        "transformer_ll_1" : losses.gaussian_nll,        
    }
    loss = loss_map.get(MODEL_NAME_FULL)   


    optimizer_map = {
        "adam" : keras.optimizers.Adam(),
        "adam_decay_001" : keras.optimizers.Adam(decay=0.01),
        "adam_decay_005" : keras.optimizers.Adam(decay=0.05),
        "adam_decay_0025" : keras.optimizers.Adam(decay=0.025),
        "custom_adam" : keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        "sgd_nesterov" : keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, decay=0.01, nesterov=True),
    }
    optimizer = optimizer_map.get(ARG_OPTIMIZER_NAME)

    metric_list = [
                   tf.keras.metrics.MeanSquaredError(),
                   tf.keras.metrics.MeanAbsolutePercentageError(),
                   tf.keras.metrics.RootMeanSquaredError()
                   ]

    model.summary()
    
    model.compile(optimizer=optimizer, 
                    loss=loss,
                    metrics=metric_list
                    )

    data_train, y_train, data_val, y_val = utils.load_dataset(ARG_MODEL_NAME, ARG_DATA_DIR, 
                type = ARG_SPEED_TYPE, is_summed = SUMMED, 
                n_timesteps = TIME_STEPS, is_test_set = False)

    #embed()
    train_dataset = tf.data.Dataset.from_tensor_slices((data_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(int(ARG_BATCH_SIZE), drop_remainder=True)
    #embed()

    history = model.fit(train_dataset, epochs=int(ARG_EPOCHS), 
                        #validation_data=(data_val, y_val), 
                        verbose=2)
    

    MODEL_OUTPUT_FOLDER = os.path.join(ARG_OUTPUT_DIR, '{}_{}_{}_{}_{}_{}_epoch={}'
                                                            .format(ARG_EXPERIMENT_NAME, MODEL_NAME, ARG_DATA_DIR,
                                                                    ARG_SPEED_TYPE,
                                                                    ARG_SUMMED,
                                                                    ARG_OPTIMIZER_NAME,
                                                                    ARG_EPOCHS))
    print(MODEL_OUTPUT_FOLDER)
    if not os.path.exists(MODEL_OUTPUT_FOLDER):
        os.makedirs(MODEL_OUTPUT_FOLDER)

    model.save("{}/{}_last-only_epoch={}.h5".format(MODEL_OUTPUT_FOLDER, ARG_EXPERIMENT_NAME, ARG_EPOCHS))


