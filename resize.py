import scipy.misc
import numpy as np
import os

def resize_flows(img_file_chunks):
    for i in range(len(img_file_chunks)):
        # as many as 1k images
        im_chunk = np.load(img_file_chunks[i]) #.astype(np.float16)
        N, H, W, C = im_chunk.shape
        resized = np.zeros((N,int(H/10), int(W/10), C))
        for j in range(im_chunk.shape[0]): # for each image in chunk
            print(im_chunk[j,:,:,:].shape)
            print(im_chunk[j,0:10,0:10,0])
            resized[j,:,:,:] = scipy.misc.imresize(im_chunk[j,:,:,:], (int(H/10), int(W/10)), interp='bilinear')
        np.save(img_file_chunks[i]+".npy", resized)
        
def crop_flows(img_file_chunks):
    for i in range(len(img_file_chunks)):
        # as many as 1k images
        im_chunk = np.load(img_file_chunks[i])
        N, H, W, C = im_chunk.shape
        resized = np.zeros((N,int(H/10), int(W/10), C)).astype(np.float16)
        for j in range(im_chunk.shape[0]): # for each image in chunk
            # take a chunk from the center and use this
            resized[j,:,:,:] = (im_chunk[j,216:264,278:342,:]).astype(np.float16)
        save_str = img_file_chunks[i][0:14]+"tmp_cropped/"+img_file_chunks[i][22:-4]+"_small.npy"
        np.save(save_str, resized)

mypath = "/home/Chelsea/gbucket"
img_files = [os.path.join(mypath, f) for f in os.listdir(mypath) if (os.path.isfile(os.path.join(mypath, f)) and (f[0] == "f"))]

crop_flows(img_files)
