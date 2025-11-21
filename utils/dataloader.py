import os
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion


class nuScenes(Dataset):
    def __init__(self, config, imageset='train', num_vote=1):
        if config.get("debug", False):
            ## If mini set is available
            # version = 'v1.0-mini'
            # scenes = splits.mini_train
            ## If mini set is not available
            if imageset == 'train' or imageset == 'val':
                version = 'v1.0-trainval'
                scenes = splits.train[:1] if imageset == 'train' else splits.val[:1]
            else:
                version = 'v1.0-test'
                scenes = splits.test[:1]
        else:
            if imageset == 'train' or imageset == 'val':
                version = 'v1.0-trainval'
                scenes = splits.train if imageset == 'train' else splits.val
            else:
                version = 'v1.0-test'
                scenes = splits.test

        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.num_vote = num_vote
        self.data_path = config['dataset_params']['train_data_loader']['data_path']
        self.imageset = imageset
        self.img_view = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 
                         'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']


        self.nusc = NuScenes(version=version, dataroot=self.data_path, verbose=True)

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
        cam_intrinsics = []
        for i in range(6):
            img, cam_token = self.loadImage(index, i)
            images.append(img)
            cam_sample_tokens.append(cam_token)
            cam_path, boxes_front_cam, cam_intrinsic = self.nusc.get_sample_data(cam_token)
            cam_intrinsics.append(cam_intrinsic)

        # Use the first camera for calibration (or extend for all cameras if needed)
        pointsensor = self.nusc.get('sample_data', lidar_sample_token)
        cs_record_lidar = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pose_record_lidar = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        
        # calculate camera extrinsics to LiDAR
        # LiDAR 2 ego transformation
        R_l = Quaternion(cs_record_lidar['rotation']).rotation_matrix
        t_l = np.array(cs_record_lidar['translation'])
        T_lidar_ego = np.eye(4)
        T_lidar_ego[:3, :3] = R_l
        T_lidar_ego[:3, 3] = t_l

        lidar2cam_list = []
        for cam_token in cam_sample_tokens:
            cam = self.nusc.get('sample_data', cam_token)
            cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
            pose_record = self.nusc.get('ego_pose', cam['ego_pose_token'])
            
            # Camera 2 ego
            R_c = Quaternion(cs_record['rotation']).rotation_matrix
            t_c = np.array(cs_record['translation'])
            T_cam_ego = np.eye(4)
            T_cam_ego[:3, :3] = R_c
            T_cam_ego[:3, 3] = t_c
            T_cam_ego_inv = np.linalg.inv(T_cam_ego)

            # LiDAR 2 Camera
            T_lidar_cam = T_cam_ego_inv @ T_lidar_ego

            lidar2cam_list.append(T_lidar_cam)

        data_dict = {
            'images': images,               # list of 6 PIL images
            'lidar_points': pointcloud,     # shape: (P, 4) [x, y, z, intensity]
            'labels': sem_label.astype(np.uint8),
            'num_points': len(pointcloud),
            'cam_intrinsic': np.array(cam_intrinsics),
            'lidar2cam_extrinsics': np.array(lidar2cam_list)
        }

        return data_dict

def create_dataloaders(config):
    """
    Create flexible DataLoaders for train, val, and test based on config.
    """

    dataloaders = {}
    # splits = ["train", "val", "test"]
    splits = ["train", "val"]

    for split in splits:
        loader_cfg = config['dataset_params'][f"{split}_data_loader"]

        batch_size  = loader_cfg['batch_size']
        shuffle     = loader_cfg['shuffle']
        num_workers = loader_cfg['num_workers']

        # Create dataset
        dataset = nuScenes(config, imageset=split)

        # Create dataloader
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=fusion_collate_fn,
        )

    return dataloaders

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
        image_sizes.append(sample['images'][0].size)
    images = torch.stack(images_list, dim=0)  # (B, 6, C, H, W)
    image_sizes = torch.tensor(image_sizes, dtype=torch.long)

    # Process LiDAR points and labels
    lidar_list = [torch.from_numpy(sample['lidar_points']).float() for sample in batch]  # (B, P, 4)
    labels_list = [torch.from_numpy(sample['labels']).long().squeeze() for sample in batch]  # (B, P)
    cam_intrinsics = [torch.from_numpy(sample['cam_intrinsic']).float().squeeze() for sample in batch]    # (B,)
    lidar2cam_extrinsics = [torch.from_numpy(sample['lidar2cam_extrinsics']).float().squeeze() for sample in batch]  # (B,)

    max_P = max([l.shape[0] for l in lidar_list])
    pcd_feat_dim = lidar_list[0].shape[1]       # 4

    lidar_points_padded = torch.zeros((B, max_P, pcd_feat_dim), dtype=torch.float32)
    labels_padded = torch.full((B, max_P), 0, dtype=torch.long)  # ignore_index: 0
    mask = torch.zeros((B, max_P), dtype=torch.bool)
    
    for i in range(B):
        P = lidar_list[i].shape[0]
        lidar_points_padded[i, :P] = lidar_list[i]
        labels_padded[i, :P] = labels_list[i]
        valid_mask = (labels_list[i] != 0)   # ignore noise class 0
        mask[i, :P] = valid_mask
    
    return images, image_sizes, lidar_points_padded, labels_padded, mask, cam_intrinsics, lidar2cam_extrinsics

def calculate_class_weights(dataloaders, device, num_classes, print_every=100):
    # class counts directly on GPU
    class_counts = torch.zeros(num_classes, device=device, dtype=torch.long)

    for i, (_, _, _, labels, mask, _, _) in enumerate(dataloaders['train']):
        labels = labels.to(device)
        mask = mask.to(device)

        # get valid labels (still on GPU)
        valid_labels = labels[mask == 1]

        # accumulate counts using bincount on GPU
        batch_count = torch.bincount(valid_labels, minlength=num_classes)
        class_counts += batch_count

        # optional periodic printing
        if (i + 1) % print_every == 0:
            weights_temp = torch.sqrt(class_counts.max() / (class_counts + 1e-6))
            print(f"[Batch {i+1}] Current class weights: {weights_temp.detach().cpu().numpy()}")

    # compute final weights on GPU
    class_weights = torch.sqrt(class_counts.max() / (class_counts + 1e-6))

    print("\n=== FINAL CLASS WEIGHTS (GPU) ===")
    print(class_weights.detach().cpu().numpy())

    return class_weights.float()

def load_class_names(config_path, use_16_classes=True):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if use_16_classes:
        class_dict = config["labels_16"]
    else:
        class_dict = config["labels"]

    # keys from YAML are strings → convert to int
    class_names = {int(k): v for k, v in class_dict.items() if int(k) != 0}     # ignore label 0 : noise

    return class_names