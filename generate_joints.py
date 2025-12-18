import codecs as cs
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin


def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


index_path = './index.csv'
save_dir = './joints'
index_file = pd.read_csv(index_path)
total_amount = index_file.shape[0]
fps = 20


for i in tqdm(range(total_amount)):
    source_path = index_file.loc[i]['source_path']
    new_name = index_file.loc[i]['new_name']
    data = np.load(source_path)
    start_frame = index_file.loc[i]['start_frame']
    end_frame = index_file.loc[i]['end_frame']
    if 'humanact12' not in source_path:
        if 'Eyes_Japan_Dataset' in source_path:
            data = data[3*fps:]
        if 'MPI_HDM05' in source_path:
            data = data[3*fps:]
        if 'TotalCapture' in source_path:
            data = data[1*fps:]
        if 'MPI_Limits' in source_path:
            data = data[1*fps:]
        if 'Transitions_mocap' in source_path:
            data = data[int(0.5*fps):]
        data = data[start_frame:end_frame]
        data[..., 0] *= -1
    
    data_m = swap_left_right(data)
#     save_path = pjoin(save_dir, )
    np.save(pjoin(save_dir, new_name), data)
    np.save(pjoin(save_dir, 'M'+new_name), data_m)