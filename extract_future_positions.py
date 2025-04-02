from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Quaternion
import numpy as np
import json
import os
from tqdm import tqdm
import multiprocessing

# Load dataset
# nusc = NuScenes(version='v1.0-trainval', dataroot='/data/sets/nuscenes', verbose=True)

def round_floats(data, decimals=3):
    """Recursively round all floats in a data structure to a given number of decimal places."""
    if isinstance(data, float):
        return round(data, decimals)
    elif isinstance(data, list):
        return [round_floats(item, decimals) for item in data]
    elif isinstance(data, dict):
        return {key: round_floats(value, decimals) for key, value in data.items()}
    return data

def cartesian_to_polar(relative_position):
    """Convert (x, y) position to (angle, distance) relative to ego vehicle."""
    x, y = relative_position[:2]  # Ignore z for angle calculations
    distance = np.linalg.norm([x, y])  # Compute Euclidean distance
    angle = np.degrees(np.arctan2(y, x))  # Compute angle in degrees
    return [round(angle, 3), round(distance, 3)]

def get_local_positions(sample_token, future_times=[1, 3, 5]):
    """Retrieve ego and object positions relative to the ego vehicle at future timestamps."""
    sample = nusc.get('sample', sample_token)
    current_time = sample['timestamp'] / 1e6  # Convert to seconds

    # Get current ego pose
    ego_pose_data = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])
    ego_translation = np.array(ego_pose_data['translation'])
    ego_rotation = Quaternion(ego_pose_data['rotation'])

    # Get current speed and heading
    velocity = np.linalg.norm(ego_translation)  # Approximate speed
    heading = np.degrees(ego_rotation.yaw_pitch_roll[0])  # Extract yaw as heading in degrees

    # Get current object positions
    current_objects = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        obj_translation = np.array(ann['translation'])
        relative_position = ego_rotation.inverse.rotate(obj_translation - ego_translation)
        current_objects.append({
            'category': ann['category_name'], 
            'position': cartesian_to_polar(relative_position)
        })

    future_ego_positions, future_objects = {}, {}
    future_sample = sample
    for future_time in future_times:
        future_time_target = current_time + future_time

        # Move forward in time until we find the closest future sample
        while future_sample['next']:
            future_sample = nusc.get('sample', future_sample['next'])
            if future_sample['timestamp'] / 1e6 >= future_time_target:
                break

        # Get future ego pose
        ego_pose_data = nusc.get('ego_pose', nusc.get('sample_data', future_sample['data']['LIDAR_TOP'])['ego_pose_token'])
        future_translation = np.array(ego_pose_data['translation'])
        relative_future_translation = ego_rotation.inverse.rotate(future_translation - ego_translation)
        future_ego_positions[future_time] = cartesian_to_polar(relative_future_translation)

        # Get future object positions
        future_objects[future_time] = []
        for ann_token in future_sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            obj_translation = np.array(ann['translation'])
            relative_position = ego_rotation.inverse.rotate(obj_translation - future_translation)
            future_objects[future_time].append({
                'category': ann['category_name'], 
                'position': cartesian_to_polar(relative_position)
            })

    return {
        "instruction": "Given the current speed, heading, and object positions, predict the future ego and object positions relative to the current ego position.",
        "input": {
            "speed": round_floats(velocity),
            "heading": round_floats(heading),
            "current_objects": current_objects
        },
        "output": {
            "future_ego_positions": future_ego_positions,
            "future_objects": future_objects
        }
    }

def process_sample(sample_token):
    """Processes a single sample and returns the result."""
    return get_local_positions(sample_token)

def process_and_save(sample_tokens):
    """Processes samples in parallel and writes output to file."""
    #num_workers = min(6, os.cpu_count() // 2)
    num_workers = 16
    
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_sample, sample_tokens), total=len(sample_tokens), desc="Processing Samples"))

    # Save results as a valid JSON array
    with open("unsloth_data.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    # Get the list of sample tokens (assuming `nusc` is initialized)
    sample_tokens = [s['token'] for s in nusc.sample]

    # Run the processing
    process_and_save(sample_tokens)

    print("Processing complete! Data saved in 'unsloth_data.json'.")
