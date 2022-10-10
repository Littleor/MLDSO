import platform

cwru_labels = ['Normal', '0.007-Roller', '0.007-InnerRace', '0.007-OuterRace', '0.014-Roller', '0.014-InnerRace',
               '0.014-OuterRace', '0.021-Roller', '0.021-InnerRace', '0.021-OuterRace']

class Config:
    mldso_config = {
        "data_type": 'CWRU',  # Dataset
        "data_speed_source": '1730',  # Speed
        "data_speed_target": '1750',
        "path": './result/MLDSO/best_model.pt',  # path
        "train_set_size": 5,  # Train set size
        "learning_rate": 0.001,
        "train_episode": 240,
        "shot": 5
    }

    public_config = {
        "data_type": 'CWRU',
        "path": './result/MLDSO/best_model.pt',
        "train_set_size": 5,
        "learning_rate": 0.001,
        "point_latitude": 2,  # TSNE
        "over_sampling_size": 400,  # Oversampling Size
        "data_length": 2048,  # Data length
        "add_snr": 0,  # Add SNR
        "data_step": 150,  # Data step
        "train_episode": 240,
        "seed": 43,
        "data_speed_source": '1730',  # Speed
        "data_speed_target": '1750',
    }

    train_config = {
        "support_set_count": 2,  # Support set size
        "query_set_count": 3,  # Query set size
        "train_episode": 200,  # Train episode
    }

    _data = {}

    def __init__(self):
        self._data_type = ''
        self._data_speed_source = ''
        self.path = ''
        self.main_train_path = ''
        self.pre_train_path = ''
        self.support_set_count = 0
        self.query_set_count = 0
        self.train_set_size = 0
        self.learning_rate = 0
        self.point_latitude = 0
        self.train_episode = 0
        self.over_sampling_size = 0
        self.data_length = 0
        self.data_step = 0
        self.add_snr = 0
        self.labels = []
        self.output_device = -1
        self.gpu_ids = []
        self.epoch = -1
        self.shot = -1
        self.seed = -1

        self.data_speed_target = None
        self.platform = platform.system().lower()

        self.data = {**self.public_config, **self.train_config}
        self.data = {**self.data, **self.mldso_config}

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.update_config()

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, data_type):
        self._data_type = data_type
        self.labels = cwru_labels 

    @property
    def data_speed_source(self):
        return self._data_speed_source

    @data_speed_source.setter
    def data_speed_source(self, data_speed_source):
        self._data_speed_source = data_speed_source
        self.labels = cwru_labels

    def update_config(self):
        self.data_type = self.data['data_type']
        self.data_speed_source = self.data['data_speed_source']
        self.path = self.data['path']
        self.support_set_count = self.data['support_set_count']
        self.query_set_count = self.data['query_set_count']
        self.train_set_size = self.data['train_set_size']
        self.learning_rate = self.data['learning_rate']
        self.point_latitude = self.data['point_latitude'] if 'point_latitude' in self.data else 2
        self.train_episode = self.data['train_episode'] if 'train_episode' in self.data else 40
        self.over_sampling_size = self.data['over_sampling_size']
        self.data_length = self.data['data_length']
        self.data_step = self.data['data_step']
        self.add_snr = self.data['add_snr']
        self.shot = self.data['shot'] if 'shot' in self.data else -1
        self.labels = cwru_labels
        self.seed = self.data['seed']

        self.data_speed_target = self.data['data_speed_target'] if 'data_speed_target' in self.data else None
        self.pre_train_path = self.data['pre_train_path'] if 'pre_train_path' in self.data else None
        self.main_train_path = self.data['main_train_path'] if 'main_train_path' in self.data else None

    def __repr__(self):
        return str(self.data)
