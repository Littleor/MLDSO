import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

from utils.main import add_noise


class DataConf:
    def __init__(self, support_set_count: int, query_set_count: int,
                 selected_labels: tuple):
        """
        :param support_set_count: Size of support set
        :param query_set_count: Size of query set
        :param selected_labels: Selected labels
        """
        self.n = len(selected_labels)
        self.support_set_count = support_set_count
        self.query_set_count = query_set_count
        self.selected_labels = selected_labels


class DataLoaderConf:
    def __init__(self, data_type='CWRU',
                 data_speed='1730', data_step=150, over_sampling_size=800, data_length=1024, train_set_size=40,
                 multiple=1, add_snr=-2):
        """
        Config of data loader
        :param data_type: CWRU or others
        :param data_speed: Speed
        :param data_step: Step
        :param over_sampling_size: Size of over sampling
        :param data_length: Length of data
        :param train_set_size: Size of train set
        :param multiple: Multiple of data
        :param add_snr: Add SNR
        """

        self.data_type = data_type
        self.data_speed = data_speed
        self.data_step = data_step
        self.data_count = over_sampling_size
        self.data_length = data_length
        self.multiple = multiple
        self.add_snr = add_snr

        self.train_max_size = train_set_size


def load_data(file_path: str, dataloader_conf: DataLoaderConf = DataLoaderConf(),
              tolist=True):
    """
    Read data from file and preprocess
    :param file_path: File path
    :param dataloader_conf: Config of data loader
    :param tolist: Convert to list
    :return: Data
    """
    # NOTE support only .mat/.xlsx/.bin file
    file_type = os.path.basename(file_path).split('.')[-1]
    support_list = ['mat', 'xlsx', 'bin']
    assert file_type in support_list, f"{file_type} not in {support_list}"

    data_x = np.zeros((0, dataloader_conf.data_length), dtype=np.float64)
    temp = np.zeros((0, dataloader_conf.data_length), dtype=np.float64)
    step = int(dataloader_conf.data_length / dataloader_conf.multiple)
    start = step * 0

    if file_type == 'mat':
        mat_dict = loadmat(file_path)
        filter_i = filter(lambda x: 'DE_time' in x, mat_dict.keys())
        filter_list = [item for item in filter_i]
        key = filter_list[0]
        time_series = mat_dict[key][:, 0]
        new_time_series = time_series[start:]
    elif file_type == 'xlsx':
        mat_dict = pd.read_excel(file_path, skiprows=1).values
        new_time_series = mat_dict[start:].flatten()
    elif file_type == 'bin':
        new_time_series = np.fromfile(file_path, dtype=float)[start:]
    else:
        return

    if dataloader_conf.add_snr:
        new_time_series = add_noise(new_time_series, dataloader_conf.add_snr)
    if dataloader_conf.data_count > 0:
        for k in range(dataloader_conf.data_count):
            sample = new_time_series[
                     k * dataloader_conf.data_step: k * dataloader_conf.data_step + dataloader_conf.data_length]
            temp = np.vstack((temp, sample))
        data_x = temp
    else:
        idx_last = -(new_time_series.shape[0] % dataloader_conf.data_length)
        clips = new_time_series[:idx_last].reshape(-1, dataloader_conf.data_length)
        data_x = np.vstack((data_x, clips))
    return data_x.tolist() if tolist else data_x


class DataSet:
    classed_data: list = []
    dataloader_conf: DataLoaderConf = None
    total_kind = -1
    data_set_path = -1
    info = {
        "processed": {}
    }

    def get_total_kind_count(self) -> int:
        """
        Get total kind count
        :return:
        """
        return len([file_name for file_name in os.listdir(self.data_set_path) if
                    (os.path.isfile(os.path.join(self.data_set_path, file_name)) and (
                            file_name.endswith('.mat') or file_name.endswith('.xlsx') or file_name.endswith('.bin')))])

    def __init__(self, dataloader_conf: DataLoaderConf, output_device=-1):
        self.dataloader_conf = dataloader_conf
        self.data_set_path = './data/' + self.dataloader_conf.data_type + '/' + self.dataloader_conf.data_speed
        self.total_kind = self.get_total_kind_count()
        self.classed_data = [[] for _ in range(self.total_kind)]
        # print(
        #         f"--- Dataset {self.dataloader_conf.data_type}(Speed:{self.dataloader_conf.data_speed}) loading ({self.dataloader_conf.train_max_size} : {self.dataloader_conf.data_count - self.dataloader_conf.train_max_size}, {self.dataloader_conf.add_snr}dB) ---")
        file_list = os.listdir(self.data_set_path)
        file_list.sort()
        for index, file_name in enumerate(file_list):
            if file_name.endswith('.mat') or file_name.endswith('.bin') or file_name.endswith('.xlsx'):
                self.classed_data[index] = load_data(f'{self.data_set_path}/{file_name}',
                                                     dataloader_conf=self.dataloader_conf)
        self.info['processed']['type_size'] = len(self.classed_data)
        self.info['processed']['size'] = [len(d) for d in self.classed_data]
        self.output_device = output_device

    def get_sample_data(self, data_conf: DataConf):
        """
        Get sample data
        :param data_conf: Data config
        :return: Sample data
        """
        support_set = [[] for _ in range(data_conf.n)]
        query_set = [[] for _ in range(data_conf.n)]

        assert len(self.classed_data[0]) > 0

        # Get support set and query set
        for data_label_index in range(data_conf.n):
            x = self.classed_data[data_label_index]  # Current category data
            # Support set must be selected from train_max_size data
            support_set_indexes = random.sample(range(self.dataloader_conf.train_max_size),
                                                data_conf.support_set_count)
            support_set[data_label_index].extend([[x[j]] for j in support_set_indexes])

            query_set_indexes = random.sample(
                    list(set(range(self.dataloader_conf.train_max_size)) - set(support_set_indexes)),
                    data_conf.query_set_count)
            query_set[data_label_index].extend([[x[j]] for j in query_set_indexes])

        support_set = torch.tensor(support_set).float()
        query_set = torch.tensor(query_set).float()
        if self.output_device != -1:
            support_set = support_set.to(self.output_device)
            query_set = query_set.to(self.output_device)
        return support_set, query_set

    def get_eval_data(self, data_conf: DataConf):
        support_set = []
        query_set = []

        for i in range(data_conf.n):
            if data_conf.query_set_count == -1:
                support_set.append(self.classed_data[i][
                                   self.dataloader_conf.train_max_size:self.dataloader_conf.train_max_size +
                                                                       data_conf.support_set_count])
                query_set.append(
                        self.classed_data[i][self.dataloader_conf.train_max_size + data_conf.support_set_count:])
            else:
                support_set.append(self.classed_data[i][
                                   self.dataloader_conf.train_max_size: self.dataloader_conf.train_max_size +
                                                                        data_conf.support_set_count])
                query_set.append(self.classed_data[i][
                                 self.dataloader_conf.train_max_size + data_conf.support_set_count:
                                 self.dataloader_conf.train_max_size + data_conf.support_set_count + data_conf.query_set_count])
        # Test
        support_set = torch.tensor(support_set).float().unsqueeze(2)
        query_set = torch.tensor(query_set).float().unsqueeze(2)
        if self.output_device != -1:
            support_set = support_set.to(self.output_device)
            query_set = query_set.to(self.output_device)
        return support_set, query_set
