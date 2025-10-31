import os
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from nuscenes import NuScenes
from nuscenes.utils import splits

class nuScenes(Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        if config.get("debug", False):
            version = 'v1.0-mini'
            scenes = splits.mini_train
        else:
            if imageset != 'test':
                version = 'v1.0-trainval'
                scenes = splits.train if imageset == 'train' else splits.val
            else:
                version = 'v1.0-test'
                scenes = splits.test

        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.num_vote = num_vote
        self.data_path = data_path
        self.imageset = imageset
        self.img_view = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 
                         'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']


        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

        self.get_available_scenes()
        available_scene_names = [s['name'] for s in self.available_scenes]
        scenes = list(filter(lambda x: x in available_scene_names, scenes))
        scenes = set([self.available_scenes[available_scene_names.index(s)]['token'] for s in scenes])
        self.get_path_infos_cam_lidar(scenes)

        print('Total %d scenes in the %s image set' % (len(self.token_list), imageset))

    def __len__(self):
        return len(self.token_list)

    def loadDataByIndex(self, index):
        lidar_sample_token = self.token_list[index]['lidar_token']
        lidar_path = os.path.join(self.data_path,
                                  self.nusc.get('sample_data', lidar_sample_token)['filename'])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            lidarseg_path = os.path.join(self.data_path,
                                         self.nusc.get('lidarseg', lidar_sample_token)['filename'])
            annotated_data = np.fromfile(lidarseg_path, dtype=np.uint8).reshape((-1, 1))

        pointcloud = raw_data[:, :4]
        sem_label = annotated_data
        inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        return pointcloud, sem_label, inst_label, lidar_sample_token

    def loadImage(self, index, image_id):
        cam_sample_token = self.token_list[index]['cam_token'][image_id]
        cam = self.nusc.get('sample_data', cam_sample_token)
        image = Image.open(os.path.join(self.nusc.dataroot, cam['filename']))
        return image, cam_sample_token

    def get_available_scenes(self):
        self.available_scenes = []
        for scene in self.nusc.scene:
            scene_token = scene['token']
            scene_rec = self.nusc.get('scene', scene_token)
            sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            lidar_path, _, _ = self.nusc.get_sample_data(sd_rec['token'])
            if Path(lidar_path).exists():
                self.available_scenes.append(scene)

    def get_path_infos_cam_lidar(self, scenes):
        self.token_list = []
        for sample in self.nusc.sample:
            scene_token = sample['scene_token']
            lidar_token = sample['data']['LIDAR_TOP']

            if scene_token in scenes:
                for _ in range(self.num_vote):
                    cam_tokens = [sample['data'][i] for i in self.img_view]
                    self.token_list.append({'lidar_token': lidar_token, 'cam_token': cam_tokens})

    def __getitem__(self, index):
        # Load LiDAR points and labels
        pointcloud, sem_label, instance_label, lidar_sample_token = self.loadDataByIndex(index)
        sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)

        # Load all 6 camera images
        images = []
        cam_sample_tokens = []
        for i in range(6):
            img, cam_token = self.loadImage(index, i)
            images.append(img)
            cam_sample_tokens.append(cam_token)

        # Use the first camera for calibration (or extend for all cameras if needed)
        cam_path, boxes_front_cam, cam_intrinsic = self.nusc.get_sample_data(cam_sample_tokens)
        pointsensor = self.nusc.get('sample_data', lidar_sample_token)
        cs_record_lidar = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pose_record_lidar = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        cam = self.nusc.get('sample_data', cam_sample_tokens)
        cs_record_cam = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pose_record_cam = self.nusc.get('ego_pose', cam['ego_pose_token'])

        calib_info = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam['translation'],
            "ego2global_rotation_cam": pose_record_cam['rotation'],
            "cam2ego_translation": cs_record_cam['translation'],
            "cam2ego_rotation": cs_record_cam['rotation'],
            "cam_intrinsic": cam_intrinsic,
        }

        data_dict = {
            'images': images,               # list of 6 PIL images
            'lidar_points': pointcloud,     # shape: (P, 4) [x, y, z, intensity]
            'labels': sem_label.astype(np.uint8),
            'calib_info': calib_info,
            'num_points': len(pointcloud)
        }

        return data_dict


def fusion_collate_fn(batch):
    """
    Collate function for nuScenes dataset with 6-camera images per sample.
    Args:
        batch: list of data_dicts from __getitem__
    Returns:
        images: (B, 6, C, H, W) float tensor
        lidar_feats: (B, max_V, 3) float tensor
        labels_padded: (B, max_V) long tensor
        mask: (B, max_V) bool tensor
    """
    B = len(batch)
    to_tensor = transforms.ToTensor()

    # Process images
    num_views = 6
    images_list = []
    image_sizes = []
    for sample in batch:
        # Convert list of PIL images to tensors and stack along view dimension
        imgs = torch.stack([to_tensor(img) for img in sample['images']], dim=0)  # (6, C, H, W)
        images_list.append(imgs)
        image_sizes.append(sample['images'][0].size())
    images = torch.stack(images_list, dim=0)  # (B, 6, C, H, W)
    image_sizes = torch.tensor(image_sizes, dtype=torch.long)

    # Process LiDAR points and labels
    lidar_list = [torch.from_numpy(sample['lidar_points']).float() for sample in batch]  # (B, P, 4)
    labels_list = [torch.from_numpy(sample['labels']).long() for sample in batch]        # (B, P, 4)

    max_P = max([l.shape[0] for l in lidar_list])
    pcd_feat_dim = lidar_list[0].shape[1]       # 4

    lidar_points_padded = torch.zeros((B, max_P, pcd_feat_dim), dtype=torch.float32)
    labels_padded = torch.full((B, max_P, pcd_feat_dim), -100, dtype=torch.long)  # ignore_index
    mask = torch.zeros((B, max_P), dtype=torch.bool)

    for i in range(B):
        P = lidar_list[i].shape[0]
        lidar_points_padded[i, :P] = lidar_list[i]
        labels_padded[i, :P] = labels_list[i]
        mask[i, :P] = 1

    return images, image_sizes, lidar_points_padded, labels_padded, mask, sample['calib_info']