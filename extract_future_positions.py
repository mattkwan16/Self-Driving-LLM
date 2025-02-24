from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Quaternion
import numpy as np
import json

# Load dataset
nusc = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)

def get_local_positions(sample_token, future_times=[1, 3, 5]):
    """Retrieve ego and object positions relative to the ego vehicle at future timestamps."""
    sample = nusc.get('sample', sample_token)
    current_time = sample['timestamp'] / 1e6  # Convert to seconds
    
    # Get current ego pose
    ego_pose_data = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])
    ego_translation = np.array(ego_pose_data['translation'])  # (x, y, z)
    ego_rotation = Quaternion(ego_pose_data['rotation'])
    
    # Get current speed (approximate from velocity vector)
    velocity_data = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['CAM_FRONT'])['ego_pose_token'])
    velocity = np.linalg.norm(np.array(velocity_data['translation']) - ego_translation)  # Speed estimation
    heading = ego_rotation.yaw_pitch_roll[0]  # Extract yaw as heading
    
    # Get current object positions
    current_objects = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        obj_translation = np.array(ann['translation'])
        
        # Convert object position to ego's local frame
        relative_position = ego_rotation.inverse.rotate(obj_translation - ego_translation)
        current_objects.append({'category': ann['category_name'], 'position': relative_position.tolist()})
    
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
    future_ego_positions = {}
    for future_time, future_sample in future_samples:
        ego_pose_data = nusc.get('ego_pose', nusc.get('sample_data', future_sample['data']['LIDAR_TOP'])['ego_pose_token'])
        future_translation = np.array(ego_pose_data['translation'])  # (x, y, z)
        future_rotation = Quaternion(ego_pose_data['rotation'])
        
        # Compute relative ego position
        relative_future_translation = ego_rotation.inverse.rotate(future_translation - ego_translation)
        future_ego_positions[future_time] = relative_future_translation.tolist()
        
        objects = []
        for ann_token in future_sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            obj_translation = np.array(ann['translation'])
            
            # Convert object position to ego's local frame
            relative_position = ego_rotation.inverse.rotate(obj_translation - future_translation)
            objects.append({'category': ann['category_name'], 'position': relative_position.tolist()})
        
        results[future_time] = objects
    
    return velocity, heading, current_objects, future_ego_positions, results

def save_to_unsloth_format(velocity, heading, current_data, future_ego, future_data, filename="unsloth_data.json"):
    """Save the results in Unsloth's Alpaca format."""
    formatted_data = {
        "Instruction": "Given the current speed, heading, and object positions, predict the future ego and object positions relative to the current ego position.",
        "Input": {"speed": velocity, "heading": heading, "current_objects": current_data},
        "Output": {"future_ego_positions": future_ego, "future_objects": future_data}
    }
    with open(filename, "w") as f:
        json.dump(formatted_data, f, indent=4)

# Example usage
sample_token = nusc.sample[0]['token']  # First sample
velocity, heading, current_positions, future_ego_positions, future_positions = get_local_positions(sample_token)
save_to_unsloth_format(velocity, heading, current_positions, future_ego_positions, future_positions)

for time, objects in future_positions.items():
    print(f"In {time} seconds:")
    print(f"  Ego vehicle relative position: {future_ego_positions[time]}")
    for obj in objects:
        print(f"  {obj['category']} at {obj['position']}")
