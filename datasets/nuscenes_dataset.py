import pickle
import numpy as np
from datasets.processor.data_processor import DataProcessor
from datasets.processor.point_feature_encoder import PointFeatureEncoder
import os

class NuScenesDataset:
    def __init__(self, dataset_cfg):
        self.root_path = dataset_cfg.get('DATA_PATH')
        self.MAX_SWEEPS = dataset_cfg.get('MAX_SWEEPS', 10)
        info_path = dataset_cfg.get('INFO_PATH', 'nuscenes_infos_10sweeps_val.pkl')
        info_path = os.path.join(self.root_path, info_path)

        with open(info_path, 'rb') as f:
            self.infos = pickle.load(f)

        self.point_cloud_range = np.array(dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_processor = DataProcessor(
            dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=False, num_point_features=self.point_feature_encoder.num_point_features
        )

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = os.path.join(self.root_path, sweep_info['lidar_path'])
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        lidar_path = os.path.join(self.root_path, info['lidar_path'])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def _get_points(self, index):
        data_dict = {'points': self.get_lidar_with_sweeps(index, self.MAX_SWEEPS)}
        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        return data_dict

if __name__ == '__main__':
    cfg_file = '/home/chenyukang/OpenPCDet-master2/tools/cfgs/dataset_configs/nuscenes_dataset.yaml'
    root_path = '/home/chenyukang/OpenPCDet-master2/data/nuscenes/v1.0-trainval'
    info_path = '/home/chenyukang/OpenPCDet-master2/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl'

    dataset_cfg = cfg_from_yaml_file(cfg_file, cfg)
    nuscenes_dataset = NuScenesDataset(
        dataset_cfg, root_path, info_path, MAX_SWEEPS=10
    )
    from IPython import embed; embed()
