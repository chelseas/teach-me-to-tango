## File that houses the data loading utilities

import numpy as np
import matplotlib.pyplot as plt

def load_data():

    ### Data Loading ###
    # [t, x_tango, y_tango, z_tango, x_vicon, y_vicon, z_vicon, x_err_rate, y_err_rate, z_err_rate]
    d_raw = (np.genfromtxt('trasnformed_data_slash_2.csv',delimiter=',')).astype(np.float32)

    pos_err = np.zeros(d_raw.shape[0])
    for k in range(d_raw.shape[0]):
        pos_err[k] = np.linalg.norm(d_raw[k,1:4]-d_raw[4:7])
    y_data = np.diff(pos_err);
    
    plt.subplot(1,2,1);
    plt.plot(pos_err)
    plt.subplot(1,2,2)
    plt.plot(y_data)
    # Timestamps from vicon/tango data
    t_raw = d_raw[:,0]
    
    # Timestamps from imu/camera data
    t_img = (np.genfromtxt('rates.csv',delimiter=',')[:,0]).astype(np.float32)
    
    # Align start of data streams
    i_start = 0
    for k in range(t_img.shape[0]):
        if(t_img[k]>t_raw[0]):
            i_start = k
            break
    if i_start is 0:
        print('Warning! Time sequence alignment failed!')

    # Extract and interpolate data
    imu_data = (np.genfromtxt('rates.csv', delimiter=',')[i_start:,1:7]).astype(np.float32)
    t_img = t_img[i_start:]    

    
    # Should really automate this... or just re-save file.
    flows_data = []
    flows_data.append(np.load('flows_lowres_1_16.npy'))
    flows_data.append(np.load('flows_lowres_2_16.npy'))
    flows_data.append(np.load('flows_lowres_3_16.npy'))
    flows_data.append(np.load('flows_lowres_4_16.npy'))
    flows_data.append(np.load('flows_lowres_5_16.npy'))
    flows_data.append(np.load('flows_lowres_6_16.npy'))
    flows_data.append(np.load('flows_lowres_7_16.npy'))
    flows_data.append(np.load('flows_lowres_8_16.npy'))
    flows_data.append(np.load('flows_lowres_9_16.npy'))
    flows_data.append(np.load('flows_lowres_10_16.npy'))
    flows_data.append(np.load('flows_lowres_11_16.npy'))
    flows_data.append(np.load('flows_lowres_12_16.npy'))
    flows_data.append(np.load('flows_lowres_13_16.npy'))
    flows_data.append(np.load('flows_lowres_14_16.npy'))
    flows_data.append(np.load('flows_lowres_15_16.npy'))
    flows_data.append(np.load('flows_lowres_16_16.npy'))
    flows_data = np.concatenate(flows_data,axis=0)
    flows_data = flows_data[i_start:,:,:,:]

    # Shorten datastream:
    i_start = 600 # remove weird samples from set-up
    i_end = 10000 # remove weird samples from set-down
    y_data = y_data[i_start:i_end,:]
    imu_data = imu_data[i_start:i_end,:]
    flows_data = flows_data[i_start:i_end,:,:,:]

    pos_err = pos_err[i_start:i_end]
    pos_err_rate = pos_err_rate[i_start:i_end]
    
    ### Data Preprocessing ###
        
    # Normalize imu data:
    imu_data = imu_data - imu_data.mean(axis=0)
    imu_data /= np.std(imu_data,axis=0)

    # Flatten flows data into a "velocity" estimate
    v = np.zeros((flows_data.shape[0],1))
    for k in range(v.shape[0]):
        v[k,0] = np.linalg.norm(flows_data[k,:,:,:])
    # Normalize image data
    v -= v.mean(axis=0)
    v /= np.std(v,axis=0)
    x_data = np.concatenate([imu_data, v],axis=1)    
    
    return x_data, flows_data, y_data

def discretize_outputs(data, n_levels):
    minval = np.min(data)
    maxval = np.max(data)
    bin_edges = np.zeros(n_levels+1)
    mean_vals  = np.zeros(n_levels)
    
    n_per_class = data.shape[0] // (n_levels)
    n_leftover  = data.shape[0] - n_levels*n_per_class
    bin_edges[0] = minval
    
    sorted_data = np.sort(data,axis=0)
    cats = np.zeros(data.shape[0])
    n_offset = 0;
    for k in range(n_levels-1):
        if(n_offset < n_leftover):
            n_offset+=1
        bin_edges[k+1] = sorted_data[n_per_class*(k+1)+n_offset]
    bin_edges[-1] = maxval
    
    for k in range(data.shape[0]):
        cats[k] = np.argmax(bin_edges > data[k])-1

    for k in range(n_levels):
        i_k = np.where(cats == k)
        if((data[i_k]).shape[0] == 0):
            mean_vals[k] = bin_edges[k]
        else:
            mean_vals[k] = np.mean(data[i_k])
        
    return cats, mean_vals

def plot_examples(x_data,flows_data,y_norm,y_angle, train_inds, val_inds):
    plt.subplot(1,2,1);
    plt.plot(x_data)
    plt.plot(train_inds, np.zeros_like(train_inds),'.')
    plt.plot(val_inds, np.zeros_like(val_inds),'o')
    plt.xlabel('Time');
    plt.ylabel('Vector inputs')
    plt.subplot(1,2,2);
    plt.plot(y_norm);
    plt.plot(train_inds, np.zeros_like(train_inds),'.')
    plt.plot(val_inds, np.zeros_like(val_inds),'o')    
    plt.xlabel('Time')
    plt.ylabel('Norm of output')
    plt.show();

    for k in range(16):
        plt.subplot(4,4,k+1)
        plt.imshow(flows_data[np.random.randint(0,y_norm.shape[0]),:,:,0]);
        plt.clim([-8,6]);
    plt.show();


def calc_rmse(predictions, targets):
    return np.sqrt(((predictions.reshape([-1]) - targets.reshape([-1])) ** 2).mean())


