## File that houses the data loading utilities

import numpy as np
import matplotlib.pyplot as plt

def load_data():
    print('Loading vector data')
    # [t, x_tango, y_tango, z_tango, x_vicon, y_vicon, z_vicon, x_err_rate, y_err_rate, z_err_rate]
    d_raw = (np.genfromtxt('trasnformed_data_slash_2.csv',delimiter=',')).astype(np.float32)
    # Align start of data streams
    t_raw = d_raw[:,0]
    t_img = (np.genfromtxt('rates.csv',delimiter=',')[:,0]).astype(np.float32)
    i_start = 0
    for k in range(t_img.shape[0]):
        if(t_img[k]>t_raw[0]):
            i_start = k
            break
    if i_start is 0:
        print('Warning! Time sequence alignment failed!')
        
    # Extract and interpolate data
    t_img = t_img[i_start:]
    imu_data = (np.genfromtxt('rates.csv', delimiter=',')[i_start:,1:7]).astype(np.float32)
    pos_tango = np.array([np.interp(t_img, t_raw, d_raw[:,1]), np.interp(t_img, t_raw, d_raw[:,2]), np.interp(t_img, t_raw, d_raw[:,3])]).T
    pos_vicon = np.array([np.interp(t_img, t_raw, d_raw[:,4]), np.interp(t_img, t_raw, d_raw[:,5]), np.interp(t_img, t_raw, d_raw[:,6])]).T
    pos_err = np.zeros(pos_tango.shape[0])
    for k in range(pos_tango.shape[0]):
        pos_err[k] = np.linalg.norm(pos_tango[k,:]-pos_vicon[k,:])
    y_data = np.concatenate([[0.],np.diff(pos_err)],axis=0); # Error rate, crudely estimated.

    print('Loading image data')
    flows_data = np.load('../gbucket/center_cropped_300x300.npy').astype(np.float32)
    flows_data = flows_data[i_start:,...]

    
    print('Preprocessing data')
    # Shorten datastream (TODO: Automate this sort of clipping by finding outliers)
    i_start = 600 # remove weird samples from set-up
    i_end = 10000 # remove weird samples from set-down
    y_data = y_data[i_start:i_end]
    imu_data = imu_data[i_start:i_end,:]
    flows_data = flows_data[i_start:i_end,...]

    ### Data Preprocessing ###
        
    # Normalize imu data:
    imu_data = imu_data - imu_data.mean(axis=0)
    imu_data /= np.std(imu_data,axis=0)

    # Flatten flows data into a "velocity" estimate
    v = np.zeros((flows_data.shape[0],1))
    for k in range(v.shape[0]):
        v[k,0] = np.linalg.norm(flows_data[k,...])
    # Normalize image data
    v -= v.mean(axis=0)
    v /= np.std(v,axis=0)
    x_data = np.concatenate([imu_data, v, d_raw[i_start:i_end,1:4]],axis=1)    
    
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

def sample_subseq(sequence_length, x, y, ret_ind = False):
    if(x.shape[0] != y.shape[0]):
        print('X and Y do not have same number of samples!', x.shape[0], y.shape[0])
    if(x.shape[0] < sequence_length):
        print('Asking for a sequence longer than the data!')
        return [],[]
    start_ind = np.random.randint(0, x.shape[0]-sequence_length);
    if(ret_ind):
        return x[start_ind:start_ind+sequence_length,...], y[start_ind:start_ind+sequence_length,...],start_ind
    return x[start_ind:start_ind+sequence_length,...], y[start_ind:start_ind+sequence_length,...]

def sample_seqbatch(batch_size, sequence_length, x, y, start_ind=None):
    if(x.shape[0] != y.shape[0]):
        print('X and Y do not have same number of samples!', x.shape[0], y.shape[0])
    if(x.shape[0] < sequence_length*batch_size):
        print('Asking for a sequence longer than the data!')
        return [],[]
    if(start_ind is None):
        start_ind = np.random.randint(0, x.shape[0]-sequence_length);
    elif(start_ind + sequence_length*batch_size > x.shape[0]):
        print('Asking for a batch/sequence/start_ind tuple that overruns')
    x_batch = x[start_ind:start_ind+sequence_length,:]
    y_batch = y[start_ind:start_ind+sequence_length,...]
    start_ind += 1
    for k in range(batch_size-1):
        start_ind += k
        x_batch = np.concatenate([x_batch, x[start_ind:start_ind+sequence_length,...]],axis=0)
        y_batch = np.concatenate([y_batch, y[start_ind:start_ind+sequence_length,...]],axis=0)
    return x_batch,y_batch
    
