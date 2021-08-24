import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from IPython import embed


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


def get_dat(dataset, dataset_y, input_dim, idxs, ids, is_summed, n_timesteps):

    M_x = np.zeros((len(idxs), n_timesteps, input_dim))
    if is_summed:
        M_y = np.zeros((len(idxs), 1))
    else:
        M_y = np.zeros((len(idxs), n_timesteps, 1))        

    #embed()

    for i, id in enumerate(idxs):
        tmp_id = np.argwhere(ids == id)
        tmp_x = dataset[tmp_id,:]
        tmp_n = len(tmp_id)
        M_x[i, (n_timesteps-tmp_n):(n_timesteps),:] = tmp_x[:,0,]
        if is_summed:        
            M_y[i, 0] = np.sum(dataset_y[tmp_id.flatten()])
        else:
            M_y[i, (n_timesteps-tmp_n):(n_timesteps),0] = dataset_y[tmp_id.flatten()]            
    return M_x, M_y


def load_dataset(model_type, 
                dataset, 
                type = "speed",
                is_summed = True, 
                n_timesteps = 500,
                is_test_set = False):
   
    np.random.seed(123) ## for reproducibility 
    PATH = "./data/" + dataset + "/"
    
    if dataset == "al":
        X_train = pd.read_csv(PATH + "X_train.csv")
        X_val = pd.read_csv(PATH + "X_val.csv")
        X_test = pd.read_csv(PATH + "X_test.csv")

        y_train = pd.read_csv(PATH + "y_train.csv")
        y_val = pd.read_csv(PATH + "y_val.csv")
        y_test = pd.read_csv(PATH + "y_test.csv")

        N_train_ids = y_train['trip_id'].to_numpy()
        N_un_train = np.unique(N_train_ids)
        N_val_ids = y_val['trip_id'].to_numpy()
        N_un_val = np.unique(N_val_ids)
        N_test_ids = y_test['trip_id'].to_numpy()
        N_un_test = np.unique(N_test_ids)

        y_train = y_train['ev_kwh']
        y_val = y_val['ev_kwh']
        y_test = y_test['ev_kwh']
        
        data_train = np.column_stack((X_train[type].to_numpy(), X_train.to_numpy()[:, 6:]))
        data_val = np.column_stack((X_val[type].to_numpy(), X_val.to_numpy()[:, 6:]))
        data_test = np.column_stack((X_test[type].to_numpy(), X_test.to_numpy()[:, 6:]))
                
        if is_summed or model_type != "dnn":
            input_dim = data_train.shape[1]

            data_train, y_train = get_dat(data_train, y_train, input_dim, N_un_train, N_train_ids, is_summed, n_timesteps)
            data_val, y_val = get_dat(data_val, y_val, input_dim, N_un_val, N_val_ids, is_summed, n_timesteps)
            data_test, y_test = get_dat(data_test, y_test, input_dim, N_un_test, N_test_ids, is_summed, n_timesteps)


    elif dataset == "pi":
        # Too small dataset for validation
        X_train = pd.read_csv(PATH + "X_train.csv")
        #X_val = pd.read_csv(PATH + "X_val.csv")
        X_test = pd.read_csv(PATH + "X_test.csv")

        y_train = pd.read_csv(PATH + "y_train.csv")
        #y_val = pd.read_csv(PATH + "y_val.csv")
        y_test = pd.read_csv(PATH + "y_test.csv")

        N_train_ids = y_train['trip_id'].to_numpy()
        N_un_train = np.unique(N_train_ids)
        #N_val_ids = y_val['trip_id'].to_numpy()
        #N_un_val = np.unique(N_val_ids)
        N_test_ids = y_test['trip_id'].to_numpy()
        N_un_test = np.unique(N_test_ids)

        y_train = y_train['ev_kwh']
        #y_val = y_val['ev_kwh']
        y_test = y_test['ev_kwh']
        
        data_train = np.column_stack((X_train[type].to_numpy(), X_train.to_numpy()[:, 6:]))
        #data_val = np.column_stack((X_val[type].to_numpy(), X_val.to_numpy()[:, 6:]))
        data_test = np.column_stack((X_test[type].to_numpy(), X_test.to_numpy()[:, 6:]))
                
        if is_summed or model_type != "dnn":
            input_dim = data_train.shape[1]

            data_train, y_train = get_dat(data_train, y_train, input_dim, N_un_train, N_train_ids, is_summed, n_timesteps)
            #data_val, y_val = get_dat(data_val, y_val, input_dim, N_un_val, N_val_ids, is_summed, n_timesteps)
            data_test, y_test = get_dat(data_test, y_test, input_dim, N_un_test, N_test_ids, is_summed, n_timesteps)

    else:
        print("Data not exist")

    data_val = np.array([])
    y_val = np.array([])
    print("Data:", data_train.shape, y_train.shape, data_val.shape, y_val.shape, data_test.shape, y_test.shape)

    if is_test_set:
        return data_test, y_test
    else:
        return data_train, y_train, data_val, y_val
