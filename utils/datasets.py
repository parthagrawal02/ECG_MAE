import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
import wfdb
from wfdb import processing
import pdb
import neurokit2 as nk
import pandas as pd

def dropna(arr, *args, **kwarg):
    assert isinstance(arr, np.ndarray)
    dropped = pd.DataFrame(arr).dropna(*args, **kwarg).values
    if arr.ndim ==  1:
        dropped = dropped.flatten()
    return dropped

class CustomDataset(Dataset):
    def __init__(self, data_path: str = "", start: int = 0, end: int = 46, sampling_rate = 250):
        self.class_map  = {
        426177001: 1,
        426783006: 2,
        164889003: 3,
        427084000: 4,
        164890007: 5,
        427393009: 6,
        426761007: 7,
        713422000: 8,
        233896004: 9,
        233897008: 0
        }
        self.data_path = data_path
        self.data = []
        y = []
        for n in range(start, end):
            for j in range(0, 10):
                for filepath in glob.iglob(self.data_path + '/WFDBRecords/' + f"{n:02}" +  '/' + f"{n:02}" + str(j) +  '/*.hea'):
                    try:
                        ecg_record = wfdb.rdsamp(filepath[:-4])
                    except Exception:
                        continue
                    # pdb.set_trace()
                    if(np.isnan(ecg_record[0]).any()):
                        print(str(filepath))
                        continue
                    numbers = re.findall(r'\d+', ecg_record[1]['comments'][2])
                    output_array = list(map(int, numbers))
                    for j in output_array: # Only classify into one of the predecided classes.
                        if int(j) in self.class_map:
                            output_array = j
                    if isinstance(output_array, list):
                        continue
                    y.append(output_array)
                    self.data.append([filepath, output_array])

    def __len__(self):

        # l3 = 0

        # for file_path, _ in self.data:
        #     ecg_signal =  wfdb.rdsamp(file_path[:-4])[0]
        #     resampled_x, _ = wfdb.processing.resample_sig(ecg_signal[:, 0], 500, 250)
        #     _, rpeaks = nk.ecg_peaks(resampled_x, sampling_rate=250, method="neurokit")
        #     l3 += len(rpeaks["ECG_R_Peaks"])

        return len(self.data)*8
    
    def __getitem__(self, idx):

        file_idx = idx //  8
        segment_idx = idx %  8
        try:
            file_path, class_name = self.data[file_idx]
            ecg_signal =  wfdb.rdsamp(file_path[:-4])[0]

            resampled_x, _ = wfdb.processing.resample_sig(ecg_signal[:, 0], 500, 250)
            _, rpeaks = nk.ecg_peaks(resampled_x, sampling_rate=500, method="neurokit")
            rpeaks["ECG_R_Peaks"] = dropna(rpeaks["ECG_R_Peaks"])

            lx = []
            n = ecg_signal.shape[1]
            for chan in range(n):
                resampled_x, _ = wfdb.processing.resample_sig(ecg_signal[:, chan], 500, 250)
                cleaned_ecg = nk.ecg_clean(resampled_x, sampling_rate=250)
                epochs = nk.ecg_segment(cleaned_ecg, rpeaks["ECG_R_Peaks"], sampling_rate=250)
                df_with_index_column = np.array(list(epochs.values())[:][segment_idx].reset_index()['index'])
                # print(df_with_index_column.shape)
                lx.append(df_with_index_column)

            lx = np.array(lx).astype(np.float32) 
            padding_length = 320 - lx.shape[1]
            lx = np.pad(lx, ((0,0), (padding_length - padding_length // 2, padding_length // 2)), 'constant', constant_values=0)
            ecg_tensor = torch.from_numpy(lx)
            print(ecg_tensor.shape)
            class_id = self.class_map[class_name]
            class_id = torch.tensor([class_id])

            return ecg_tensor, class_id
        except:
            print("Exception", idx)
            return self.__getitem__(idx +  1)
            # file_idx = 0
            # segment_idx = 1
            # file_path, class_name = self.data[file_idx]
            # ecg_signal =  wfdb.rdsamp(file_path[:-4])[0]

            # resampled_x, _ = wfdb.processing.resample_sig(ecg_signal[:, 0], 500, 250)
            # _, rpeaks = nk.ecg_peaks(resampled_x, sampling_rate=500, method="neurokit")
            # rpeaks["ECG_R_Peaks"] = dropna(rpeaks["ECG_R_Peaks"])

            # lx = []
            # n = ecg_signal.shape[1]
            # for chan in range(n):
            #     resampled_x, _ = wfdb.processing.resample_sig(ecg_signal[:, chan], 500, 250)
            #     cleaned_ecg = nk.ecg_clean(resampled_x, sampling_rate=250)
            #     epochs = nk.ecg_segment(cleaned_ecg, rpeaks["ECG_R_Peaks"], sampling_rate=250)
            #     df_with_index_column = np.array(list(epochs.values())[:][segment_idx].reset_index()['index'])
            #     # print(df_with_index_column.shape)
            #     lx.append(df_with_index_column)

            # lx = np.array(lx).astype(np.float32) 
            # padding_length = 320 - lx.shape[1]
            # lx = np.pad(lx, ((0,0), (padding_length - padding_length // 2, padding_length // 2)), 'constant', constant_values=0)
            # ecg_tensor = torch.from_numpy(lx)
            # print(ecg_tensor.shape)
            # class_id = self.class_map[class_name]
            # class_id = torch.tensor([class_id])

            # return ecg_tensor, class_id