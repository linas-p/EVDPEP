import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


def get_for_plot(X, y):
    tim = X['seconds']
    sp = X['speed']
    lim = X['speed_limit']
    eu = y['ev_kwh']
    avg = X['speed_avg']
    sp2 = X['speed_avg_week_time']
    return tim, sp, lim, eu, avg, sp2


def to_weekend(vals):
    weekDays = (0,0,0,0,0,1,1)
    values = []
    for v in vals:
        thisXMas    = datetime.date(int(str(v)[:4]), int(str(v)[4:6]), int(str(v)[6:]))
        thisXMasDay = thisXMas.weekday()
        thisXMasDayAsString = weekDays[thisXMasDay]
        values.append(thisXMasDayAsString)
    values = np.array(values)
    return values


def get_dat(dataset, dataset_y, input_dim, idxs, ids, n_timesteps):

    M_x = np.zeros((len(idxs), n_timesteps, input_dim))
    M_y = np.zeros((len(idxs), n_timesteps, 1))


    for i, id in enumerate(idxs):
        tmp_id = np.argwhere(ids == id)
        tmp_x = dataset[tmp_id,:]
        tmp_n = len(tmp_id)
        M_x[i, (n_timesteps-tmp_n):(n_timesteps),:] = tmp_x[:,0,]
        M_y[i, (n_timesteps-tmp_n):(n_timesteps),0] = dataset_y[tmp_id.flatten()]
    return M_x, M_y


def load_dataset(model_type, input_dim, X_train, ytrain, X_val, yval, X_test, ytest,
                N_un_train, N_train_ids, N_un_val, N_val_ids,
                N_un_test, N_test_ids, n_timesteps):
    
   
    np.random.seed(123) ## rezultatų atkartojimui
    
    
    if model_type == "dnn":
        data_train = X_train
        y_train    = ytrain
        data_val   = X_val
        y_val      = yval
        data_test  = X_test
        y_test     = ytest
    elif model_type == "lstm" or model_type == "transformer":
        data_train, y_train = get_dat(X_train, ytrain, input_dim, N_un_train, N_train_ids, n_timesteps)
        data_val, y_val = get_dat(X_val, yval, input_dim, N_un_val, N_val_ids, n_timesteps)
        data_test, y_test = get_dat(X_test, ytest, input_dim, N_un_test, N_test_ids, n_timesteps)

    if model_type == "transformer":
        data_train = np.expand_dims(data_train, axis = 3)
        data_val = np.expand_dims(data_val, axis = 3)
        data_test = np.expand_dims(data_test, axis = 3)

    #if model_type == "transformer":
    #    data_train = np.expand_dims(data_train, axis = 1)
    #    data_val = np.expand_dims(data_val, axis = 1)
    #    data_test = np.expand_dims(data_test, axis = 1)    

    
    print(data_train.shape, y_train.shape, data_val.shape, y_val.shape, data_test.shape, y_test.shape)

    return data_train, y_train, data_val, y_val, data_test, y_test