from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Quaternion
import numpy as np

# Load dataset
nusc = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)

def get_local_positions(sample_token, future_times=[1, 3, 5]):
    """Retrieve ego and object positions relative to the ego vehicle at future timestamps."""
    sample = nusc.get('sample', sample_token)
    current_time = sample['timestamp'] / 1e6  # Convert to seconds
    
    future_samples = []
    for future_time in future_times:
        future_sample = sample
        future_time_target = current_time + future_time
        
        while future_sample['next']:
            future_sample = nusc.get('sample', future_sample['next'])
            future_sample_time = future_sample['timestamp'] / 1e6
            
            if future_sample_time >= future_time_target:
                future_samples.append((future_time, future_sample))
                break
    
    results = {}
    for future_time, future_sample in future_samples:
        ego_pose_data = nusc.get('ego_pose', nusc.get('sample_data', future_sample['data']['LIDAR_TOP'])['ego_pose_token'])
        ego_translation = np.array(ego_pose_data['translation'])  # (x, y, z)
        ego_rotation = Quaternion(ego_pose_data['rotation'])
        
        objects = []
        for ann_token in future_sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            obj_translation = np.array(ann['translation'])
            
            # Convert object position to ego's local frame
            relative_position = ego_rotation.inverse.rotate(obj_translation - ego_translation)
            objects.append({'category': ann['category_name'], 'position': relative_position})
        
        results[future_time] = objects
    
    return results

# Example usage
sample_token = nusc.sample[0]['token']  # First sample
local_positions = get_local_positions(sample_token)
for time, objects in local_positions.items():
    print(f"In {time} seconds:")
    for obj in objects:
        print(f"  {obj['category']} at {obj['position']}")
