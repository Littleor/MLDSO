import random
from typing import Optional, Any, Tuple

import matplotlib.colors as mcolors
import pandas as pd
import seaborn as seaborn
import torch.nn
import torch.nn.functional as F
from easydl import *
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from torch.autograd import Function

from config import Config


def get_confusion_matrix(true_labels, pred_labels):
    if pred_labels.device != torch.device('cpu'):
        pred_labels = pred_labels.to(torch.device('cpu'))
    if true_labels.device != torch.device('cpu'):
        true_labels = true_labels.to(torch.device('cpu'))
    return confusion_matrix(true_labels, pred_labels)


def seed_init(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_init(config: Config, writer):
    seed_init(config.seed)
    if torch.cuda.is_available():
        gpus = 1
        config.gpu_ids = select_GPUs(gpus, max_utilization=1.0, max_memory_usage=1.0)
        config.output_device = config.gpu_ids[0]

        print("GPU: ", config.output_device)
        torch.cuda.set_device(config.output_device)
    config_text = f"MLDSO: From {config.data_speed_source} to {config.data_speed_target}, {config.shot}-Shot"
    config_text = config_text + f'''
    Main config：
    \tDataset: {config.data_type}, Speed: {config.data_speed_source}
    \tDataset size:{config.train_set_size},Support set size: {config.support_set_count} Query set size: {config.query_set_count}
    \tTotal Size: {config.over_sampling_size}, Noise: {config.add_snr}
    \tLearning rate: {config.learning_rate}, Number of categories: {len(config.labels)}
    Other config：
    \tSampling length: {config.data_length}, Sampling step: {config.data_step}, Train episode: {config.train_episode}
    \tCUDA：{torch.cuda.is_available() and config.output_device != -1}{f',Device: ({torch.cuda.current_device()}) {torch.cuda.get_device_name()}' if torch.cuda.is_available() and config.output_device != -1 else ''}, Writer: {writer.log_dir if writer else 'None'} 
    '''
    print(config_text)
    if writer:
        writer.add_text('Config/Info', config_text)


def draw_tsne(points, type_count, path: str = None, show_image=False):
    plt.close()
    if points is None:
        return None
    classes = [i for i in range(type_count)]
    x = [[] for _ in range(type_count)]
    y = [[] for _ in range(type_count)]
    for point in points:
        x[int(point[2])].append(point[0])
        y[int(point[2])].append(point[1])
    colors = list(mcolors.TABLEAU_COLORS.keys())
    plt.rcParams['font.sans-serif'] = ['Songti SC']
    plt.rcParams['figure.figsize'] = 9.6, 7.2

    for i in range(type_count):
        plt.scatter(x[i], y[i], c=mcolors.TABLEAU_COLORS[colors[i]])
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if path is not None:
        plt.savefig(path, format='svg')
    fig = plt.gcf()
    if show_image:
        plt.show()
    return fig


def get_all_example(support_set: torch.Tensor, query_set: torch.Tensor):
    """
    Get all examples from support set and query set
    :param support_set:
    :param query_set:
    :return:
    """
    n_class = support_set.size()[0]
    assert query_set.size()[0] == n_class

    n_support_example = support_set.size()[1]
    n_query_example = query_set.size()[1]

    return torch.cat([support_set.view(n_class * n_support_example, *support_set.size()[2:]),
                      query_set.view(n_class * n_query_example, *query_set.size()[2:])], 0)


def get_loss(network: nn.Module, support_set: torch.Tensor, query_set: torch.Tensor, center_loss=None,
             point_latitude=False):
    """
    Get loss from network
    :param network: network
    :param support_set: support set
    :param query_set:   query set
    :param point_latitude: whether to use point latitude
    :return:
    """
    all_example = get_all_example(support_set, query_set)

    feature = network(all_example)

    y_hat, loss, res, points = get_triplet_margin_loss_with_feature(feature, support_set, query_set,
                                                                    center_loss=center_loss,
                                                                    point_latitude=point_latitude,
                                                                    is_train=network.training)
    return y_hat, loss, res, points, all_example


def get_triplet_margin_loss_with_feature(feature: torch.Tensor, support_set: torch.Tensor, query_set: torch.Tensor,
                                         center_loss=None,
                                         point_latitude=False, is_train=True):
    points = None

    n_class = support_set.size()[0]
    n_support_example = support_set.size()[1]
    n_query_example = query_set.size()[1]
    _, target_indexes = meta_dataset_get_labels(n_class, n_support_example, n_query_example)
    target_indexes = target_indexes.view(n_class, n_query_example, 1)

    support_lables: torch.Tensor = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_support_example,
                                                                                       1).long()
    if torch.cuda.is_available():
        support_lables = support_lables.to(torch.cuda.current_device())

    target_indexes.requires_grad = False
    if torch.cuda.is_available() and support_set.device != torch.device('cpu'):
        target_indexes = target_indexes.to(torch.cuda.current_device())
    if support_set.device == torch.device('cpu'):
        target_indexes = target_indexes.to(torch.device('cpu'))

    z_dim = feature.size()[-1]

    support_set_feature = feature[:n_class * n_support_example]
    query_set_feature = feature[n_class * n_support_example:]
    z_proto = support_set_feature.view(n_class, n_support_example, z_dim).mean(
            1)
    dists: torch.Tensor = euclidean_dist(query_set_feature, z_proto)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query_example, -1)
    value_max, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, target_indexes.squeeze(dim=2)).float().mean()

    total_loss = None
    if is_train:
        if n_support_example + n_query_example <= 2:
            # One-shot
            proto_dists: torch.Tensor = euclidean_dist(z_proto, z_proto)
            _, indices = dists.max(0)
            for i in range(proto_dists.size(-1)):
                # Replace 0 with the maximum to avoid wrong selection
                proto_dists[i][i] = proto_dists[i][indices[i]]
            proto_log_p_y = F.log_softmax(proto_dists, dim=1)
            _, index = proto_dists.min(0)
            total_loss = -proto_log_p_y.gather(-1, index.view(n_class, 1)).mean()
            print("One-shot loss", total_loss)
        else:
            # Triptlet loss
            for i in range(n_class):
                _, indices = dists[i * n_query_example: (i + 1) * n_query_example].max(0)

                max_dist_sample_index = indices[i]  # The index of the sample with the maximum distance
                positive = z_proto[i]
                anchor = torch.cat((query_set_feature[i * n_query_example: (i + 1) * n_query_example],
                                    support_set_feature[i * n_support_example: (i + 1) * n_support_example]))
                dist = dists[i * n_query_example: (i + 1) * n_query_example][
                    max_dist_sample_index].cpu().detach().numpy()
                dist[i] = np.max(dist)
                negative_proto_index = np.argmin(dist)
                negative = torch.cat((query_set_feature[
                                      negative_proto_index * n_query_example: (
                                                                                      negative_proto_index + 1) * n_query_example],
                                      support_set_feature[negative_proto_index * n_support_example: (
                                                                                                            negative_proto_index + 1) * n_support_example]))
                loss = torch.nn.functional.triplet_margin_loss(anchor, positive.view(1, -1), negative, margin=20)
                if total_loss == None:
                    total_loss = loss / n_class
                else:
                    total_loss += loss / n_class
            # Prototype clustering loss
            if center_loss != None and n_support_example + n_query_example > 2:
                current_center_loss = center_loss(z_proto, feature,
                                                  torch.cat(
                                                          (support_lables.reshape(-1),
                                                           target_indexes.reshape(-1))).reshape(
                                                          -1)
                                                  )
                total_loss = current_center_loss * (1 - 0.99995) + total_loss * 0.99995
    # TSNE
    if point_latitude and point_latitude > 0:
        points = []
        tsne = manifold.TSNE(n_components=point_latitude)
        if torch.cuda.is_available() and support_set.device != torch.device('cpu'):
            p = tsne.fit_transform(query_set_feature.detach().cpu().numpy())  # Size([nq, point_latitude])
            labels = y_hat.view(-1, 1).detach().cpu().numpy()  # Size([nq, 1])
        else:
            p = tsne.fit_transform(query_set_feature.detach().numpy())  # Size([nq, point_latitude])
            labels = y_hat.view(-1, 1).detach().numpy()  # Size([nq, 1])
        points.extend(np.hstack((p, labels)).tolist())

    return y_hat, total_loss, {
        'loss': -1 if total_loss is None else total_loss.item(),
        'acc': acc_val.item()
    }, points


def euclidean_dist(x, prototype):
    n = x.size(0)
    m = prototype.size(0)
    d = x.size(1)
    assert d == prototype.size(1)
    x = x.unsqueeze(1).expand(n, m, d)  # Size([N, M, D])
    prototype = prototype.unsqueeze(0).expand(n, m, d)  # Size([N, M, D])

    return torch.pow(x - prototype, 2).sum(2)  # Size([N, M])


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


def calculate_accuracy(feature_extractor, support_set, query_set, point_latitude=False, use_triple=True):
    """
    Calculate the accuracy of the model
    :param feature_extractor: network
    :param support_set: support set
    :param query_set: query set
    :return: accuracy
    """
    feature_extractor.eval()
    y_hat, loss_val, res, points, all_example = get_loss(feature_extractor, support_set, query_set,
                                                         point_latitude=point_latitude)
    feature_extractor.train()
    return {
        'acc': res['acc'],
        'pred': y_hat,
        "loss": loss_val,
        'points': points
    }


# N way K shot
def meta_dataset_get_labels(n_way: int, support_size: int, query_size: int):
    get_labels = lambda size: torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, size,
                                                                              1).long().reshape(-1)
    support_labels: torch.Tensor = get_labels(support_size)
    query_labels: torch.Tensor = get_labels(query_size)
    support_labels.requires_grad = False
    query_labels.requires_grad = False

    return support_labels, query_labels


def add_noise(x, snr):
    P_signal = np.sum(abs(x) ** 2) / len(x)
    P_noise = P_signal / 10 ** (snr / 10.0)
    return x + np.random.randn(len(x)) * np.sqrt(P_noise)
