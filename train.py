import sys

import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from network import MLDSONetwork

sys.path.append(".")  # import config

from utils.center_loss import CenterLoss
from config import Config
from dataset import DataSet, DataConf, DataLoaderConf
from utils.main import draw_tsne, get_loss, calculate_accuracy, config_init


def train_network(local_network: nn.Module, local_dataset: DataSet, episode: int, tag: str = '',
                  is_pretrain=True):
    global source_dataset

    local_network.train()
    center_loss = CenterLoss(num_classes=len(config.labels), feat_dim=8192, use_gpu=config.output_device != -1)
    params = list(local_network.parameters()) + list(center_loss.parameters())
    optimizer = optim.Adam(params, lr=config.learning_rate)
    support_set_size, query_set_size = (config.support_set_count, config.query_set_count) if is_pretrain else (
        config.shot // 2, config.shot - (config.shot // 2))

    max_acc = 0
    min_loss = 10000

    for turn in trange(episode, desc=f"{'Pretrain' if is_pretrain else 'FineTune'} - {tag}",
                       ascii=config.platform == 'windows'):
        if config.shot <= 1 and not is_pretrain:
            _, support_set = local_dataset.get_sample_data(
                    data_conf=DataConf(0, config.shot, selected_labels=config.labels))
            _, query_set = source_dataset.get_sample_data(
                    data_conf=DataConf(config.support_set_count, config.query_set_count, selected_labels=config.labels))
        else:
            support_set, query_set = local_dataset.get_sample_data(
                    data_conf=DataConf(support_set_size, query_set_size, selected_labels=config.labels))
        _, loss, res, _, all_example = get_loss(local_network, support_set, query_set,
                                                center_loss=center_loss,
                                                point_latitude=False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if max_acc <= res['acc'] and min_loss >= res['loss']:
            torch.save(local_network.state_dict(), f'{run_dir}/{"best_model" if is_pretrain else "eval_model"}.pt')
            max_acc = res['acc']
            min_loss = res['loss']

        writer.add_scalar(f"{'PreTrain' if is_pretrain else 'FineTune'}/Loss", res['loss'], turn)
        writer.add_scalar(f"{'PreTrain' if is_pretrain else 'FineTune'}/Accuracy", res['acc'], turn)
    print(f"{'PreTrain' if is_pretrain else 'FineTune'} Train Max Accuracy: {max_acc} Train Min Loss: {min_loss}")


def train(episode: int, tag: str = ''):
    global network, source_dataset
    train_network(network, source_dataset, episode, tag, is_pretrain=True)


def clear_feature_extractor(*args, **kwargs):
    global network
    if config.output_device == -1:
        network = MLDSONetwork().train(True)
    else:
        network = nn.DataParallel(MLDSONetwork(),
                                  device_ids=config.gpu_ids,
                                  output_device=config.output_device).train(True)


def eval_feature_extractor(tag: str):
    global eval_dataset, network
    clear_feature_extractor()
    # Load model
    network.load_state_dict(torch.load(f'{run_dir}/best_model.pt'))
    # Fine-tune
    train_network(network, eval_dataset, episode=config.train_episode, tag=tag, is_pretrain=False)
    # Test
    res = calculate_accuracy(network, eval_support_set, eval_query_set, point_latitude=2)

    tsne_fig = draw_tsne(res['points'], len(config.labels),
                         path=f'{run_dir}/TSNE/TSNE-{tag}.svg',
                         show_image=False)
    if tsne_fig is not None:
        writer.add_figure(f'TSNE/{tag}', tsne_fig)
    writer.add_text(f'TSNE/{tag}/Point', str(res['points']))
    print(f'{tag}: Test Acc', res['acc'])


def main(set_config, tag: str, path=None):
    global config, run_dir, writer, network, source_dataset, eval_dataset, eval_support_set, eval_query_set, result_acc
    config = set_config
    run_dir = f'./result/MLDSO/{tag}'
    result_acc = []
    if path is None:
        writer = SummaryWriter(log_dir=run_dir)  # Tensorboard
    else:
        writer = None
    config_init(config, writer)
    clear_feature_extractor()
    source_dataset = DataSet(
            DataLoaderConf(data_type=config.data_type, data_speed=config.data_speed_source,
                           data_step=config.data_step,
                           over_sampling_size=config.over_sampling_size, data_length=config.data_length,
                           train_set_size=config.train_set_size, add_snr=config.add_snr),
            output_device=config.output_device)
    eval_dataset = DataSet(
            DataLoaderConf(data_type=config.data_type, data_speed=config.data_speed_target,
                           data_step=config.data_step,
                           over_sampling_size=config.over_sampling_size, data_length=config.data_length,
                           train_set_size=config.train_set_size,
                           add_snr=config.add_snr),
            output_device=config.output_device)
    eval_support_set, eval_query_set = eval_dataset.get_eval_data(
            data_conf=DataConf(config.shot, -1,
                               selected_labels=config.labels))
    clear_feature_extractor()
    train(config.train_episode, tag)
    eval_feature_extractor(tag)
    return result_acc


if __name__ == '__main__':
    result_acc = None
    config = Config()
    main(config, tag='MLDSO Demo')
