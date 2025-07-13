# -*- coding: utf-8 -*-

import cv2
import operator
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema

 
def smooth(x, window_len=13, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
        
    example:
    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
 
    if window == 'flat':  
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]
 

class Frame:
    """class to hold information about each frame
    
    """
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff
 
    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id
 
    def __gt__(self, other):
        return other.__lt__(self)
 
    def __eq__(self, other):
        return self.id == other.id and self.id == other.id
 
    def __ne__(self, other):
        return not self.__eq__(other)
 
 
def rel_change(a, b):
   x = (b - a) / max(a, b)
   print(x)
   return x
 
def extract_keyframes(frame_path, 
                      save_path=None, 
                      save_flag=False,
                      len_window=50,
                      NUM_TOP_FRAMES=50,
                      THRESH=0.6,
                      USE_TOP_ORDER=False,
                      USE_THRESH=False,
                      USE_LOCAL_MAXIMA=True,
                      ):
    
    if save_flag:
        os.makedirs(save_path, exist_ok=True)
        print("frame save directory: " + save_path)

    #smoothing window size
    len_window = len_window
    # load video and compute diff between frames
    curr_frame = None
    prev_frame = None 
    frame_diffs = []
    frames = []
    i = 0 
    for file in os.listdir(frame_path):
        if not file.endswith('.jpg'):
            continue
        frame = cv2.imread(os.path.join(frame_path, file))
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        if curr_frame is not None and prev_frame is not None:
            #logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            frames.append(frame)
        prev_frame = curr_frame
        i = i + 1
    print("frame_num:{}",i) 
    
    # compute keyframe
    keyframe_id_set = set()

    if USE_TOP_ORDER:
        # sort the list in descending order
        frames.sort(key=operator.attrgetter("diff"), reverse=True)
        for keyframe in frames[:NUM_TOP_FRAMES]:
            keyframe_id_set.add(keyframe.id) 
    if USE_THRESH:
        print("Using Threshold")
        for i in range(1, len(frames)):
            if (rel_change(np.float64(frames[i - 1].diff), np.float64(frames[i].diff)) >= THRESH):
                keyframe_id_set.add(frames[i].id)   
    if USE_LOCAL_MAXIMA:
        print("Using Local Maxima")
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_window)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        for i in frame_indexes:
            keyframe_id_set.add(frames[i - 1].id)
            
        if save_flag:
            plt.figure(figsize=(40, 20))
            # plt.locator_params(numticks=100)
            plt.locator_params()
            plt.stem(sm_diff_array)
            plt.savefig(save_path + 'plot.png')
    
    # save all keyframes as image
    curr_frame = None
    keyframes_id = []
    idx=0
    
    for file in os.listdir(frame_path):
        if not file.endswith('.jpg'):
            continue
        
        if idx in keyframe_id_set:
            name = "keyframe_" + str(idx) + ".jpg"
            if save_flag:
                frame = cv2.imread(os.path.join(frame_path, file))
                cv2.imwrite(save_path + name, frame)
            keyframe_id_set.remove(idx)
            keyframes_id.append(idx)
        idx = idx + 1
    return keyframes_id