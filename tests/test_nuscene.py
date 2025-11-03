from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='/scratch/rob535f25s001_class_root/rob535f25s001_class/pohsun/datasets/nuscenes', verbose=True)
print(nusc)