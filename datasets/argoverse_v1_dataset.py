# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import TemporalData


class ArgoverseV1Dataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(ArgoverseV1Dataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        am = ArgoverseMap()
        for raw_path in tqdm(self.raw_paths):
            kwargs = process_argoverse(self._split, raw_path, am, self._local_radius)
            data = TemporalData(**kwargs)
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])


def process_argoverse(split: str,
                      raw_path: str,
                      am: ArgoverseMap,
                      radius: float) -> Dict:
    df = pd.read_csv(raw_path)

    # filter out actors that are unseen during the historical time steps
    timestamps = list(np.sort(df['TIMESTAMP'].unique())) #所有的时间戳
    historical_timestamps = timestamps[: 20]    #前20个时间戳
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)] #前20个时间戳的数据
    actor_ids = list(historical_df['TRACK_ID'].unique())    #前20个时间戳的数据中的所有actor_id, unique()去重
    df = df[df['TRACK_ID'].isin(actor_ids)] #在actor_ids中的数据
    num_nodes = len(actor_ids) #前20个时间戳中的actor数量

    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc #获取object_type为AV的数据
    av_index = actor_ids.index(av_df[0]['TRACK_ID']) #拿到第一个AV的index
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc #获取object_type为AGENT的数据
    agent_index = actor_ids.index(agent_df[0]['TRACK_ID']) #拿到第一个AGENT的index
    city = df['CITY_NAME'].values[0] #获取城市名

    # make the scene centered at AV
    origin = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float) #获取第20个时间戳的AV的坐标
    av_heading_vector = origin - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float) #获取第19个时间戳的AV的坐标
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0]) #计算第19个时间戳的AV的坐标与第20个时间戳的AV的坐标的夹角
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])#计算第19个时间戳的AV的坐标与第20个时间戳的AV的坐标的旋转矩阵

    # initialization
    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float) #
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # torch.t()转置，contiguous()返回一个内存连续的有相同数据的tensor
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous() #生成所有边全排列组合
    """edge_index=[01,
        12,
        23,
        ...]
    """
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool) # actor_num * 50, bool
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool) # actor_num * 20, bool
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float) # actor_num, float

    for actor_id, actor_df in df.groupby('TRACK_ID'): #相同Track_id的数据，迭代访问它的actor_id, actor_df
        node_idx = actor_ids.index(actor_id) # actor_id在actor_ids中的index
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']] #获取actor_id的所有时间戳的index
        padding_mask[node_idx, node_steps] = False #
        if padding_mask[node_idx, 19]:  # make no predictions for actors that are unseen at the current time step
            padding_mask[node_idx, 20:] = True
        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float() #获取actor_id的所有时间戳的坐标
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat) #将actor_id的所有时间戳的坐标转换为以第20个时间戳的AV为原点的坐标系
        node_historical_steps = list(filter(lambda node_step: node_step < 20, node_steps)) #获取actor_id的所有时间戳的index中小于20的index
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately) 
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]] #计算actor_id的最后一个时间戳的坐标与倒数第二个时间戳的坐标的向量
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0]) #计算actor_id的最后一个时间戳的坐标与倒数第二个时间戳的坐标的向量的夹角
        else:  # make no predictions for the actor if the number of valid time steps is less than 2 #如果actor_id的所有时间戳的index中小于20的index的数量小于2，则不进行预测
            padding_mask[node_idx, 20:] = True #将actor_id的所有时间戳的index中大于等于20的index的mask设为True

    # bos_mask is True if time step t is valid and time step t-1 is invalid #如果时间步t有效且时间步t-1无效，则bos_mask为True
    bos_mask[:, 0] = ~padding_mask[:, 0] #将padding_mask的第一列取反，赋值给bos_mask的第一列
    bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20] #将padding_mask的第一列到第20列取反，赋值给bos_mask的第一列到第20列

    positions = x.clone() #将x的值赋值给positions
    x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 30, 2),
                            x[:, 20:] - x[:, 19].unsqueeze(-2)) #将x的第20列到最后一列的值减去x的第19列的值，赋值给x的第20列到最后一列
    x[:, 1: 20] = torch.where((padding_mask[:, : 19] | padding_mask[:, 1: 20]).unsqueeze(-1),
                              torch.zeros(num_nodes, 19, 2),
                              x[:, 1: 20] - x[:, : 19]) #将x的第1列到第19列的值减去x的第0列到第18列的值，赋值给x的第1列到第19列
    x[:, 0] = torch.zeros(num_nodes, 2) #将x的第0列的值设为0

    # get lane features at the current time step #获取当前时间步的车道特征
    df_19 = df[df['TIMESTAMP'] == timestamps[19]] #获取时间戳为timestamps[19]的数据
    node_inds_19 = [actor_ids.index(actor_id) for actor_id in df_19['TRACK_ID']] #获取时间戳为timestamps[19]的数据的所有actor_id在actor_ids中的index
    node_positions_19 = torch.from_numpy(np.stack([df_19['X'].values, df_19['Y'].values], axis=-1)).float() #获取时间戳为timestamps[19]的数据的所有坐标
    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
     lane_actor_vectors) = get_lane_features(am, node_inds_19, node_positions_19, origin, rotate_mat, city, radius) #获取时间戳为timestamps[19]的数据的所有车道特征

    y = None if split == 'test' else x[:, 20:] #如果split为test，则y为None，否则y为x的第20列到最后一列
    seq_id = os.path.splitext(os.path.basename(raw_path))[0] #获取raw_path的文件名

    return {
        'x': x[:, : 20],  # [N, 20, 2] #x的第0列到第19列
        'positions': positions,  # [N, 50, 2] #
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y,  # [N, 30, 2]
        'num_nodes': num_nodes, #
        'padding_mask': padding_mask,  # [N, 50]
        'bos_mask': bos_mask,  # [N, 20]
        'rotate_angles': rotate_angles,  # [N]
        'lane_vectors': lane_vectors,  # [L, 2]
        'is_intersections': is_intersections,  # [L]
        'turn_directions': turn_directions,  # [L]
        'traffic_controls': traffic_controls,  # [L]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
        'seq_id': int(seq_id),
        'av_index': av_index,
        'agent_index': agent_index,
        'city': city,
        'origin': origin.unsqueeze(0),
        'theta': theta, 
    } 


def get_lane_features(am: ArgoverseMap,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    for node_position in node_positions: 
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius)) #获取node_position附近的车道id
    node_positions = torch.matmul(node_positions - origin, rotate_mat).float() #将node_positions减去origin，乘以rotate_mat，赋值给node_positions
    for lane_id in lane_ids: 
        lane_centerline = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, : 2]).float() #获取车道id为lane_id的车道的中心线
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat) #将lane_centerline减去origin，乘以rotate_mat，赋值给lane_centerline
        is_intersection = am.lane_is_in_intersection(lane_id, city) #判断车道id为lane_id的车道是否在交叉口
        turn_direction = am.get_lane_turn_direction(lane_id, city) #获取车道id为lane_id的车道的转向
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city) #判断车道id为lane_id的车道是否有交通管制
        lane_positions.append(lane_centerline[:-1]) #将lane_centerline的第0行到倒数第2行的值赋值给lane_positions
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1]) #将lane_centerline的第1行到最后一行的值减去lane_centerline的第0行到倒数第2行的值，赋值给lane_vectors
        count = len(lane_centerline) - 1 #获取lane_centerline的行数-1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8)) #将is_intersection乘以count，赋值给is_intersections
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8)) #将turn_direction乘以count，赋值给turn_directions
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8)) #将traffic_control乘以count，赋值给traffic_controls
    lane_positions = torch.cat(lane_positions, dim=0) #将lane_positions按列拼接
    lane_vectors = torch.cat(lane_vectors, dim=0) #将lane_vectors按列拼接
    is_intersections = torch.cat(is_intersections, dim=0) #将is_intersections按列拼接
    turn_directions = torch.cat(turn_directions, dim=0) #将turn_directions按列拼接
    traffic_controls = torch.cat(traffic_controls, dim=0) #将traffic_controls按列拼接

    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous() #获取lane_vectors的行数和node_inds的行数的笛卡尔积
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1) #将lane_positions按列重复len(node_inds)次，减去node_positions按行重复lane_vectors.size(0)次，赋值给lane_actor_vectors
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius #获取lane_actor_vectors的二范数小于radius的mask
    lane_actor_index = lane_actor_index[:, mask] #获取lane_actor_index的第0行到最后一行，mask为True的列，赋值给lane_actor_index
    lane_actor_vectors = lane_actor_vectors[mask] #获取lane_actor_vectors，mask为True的行，赋值给lane_actor_vectors

    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors
