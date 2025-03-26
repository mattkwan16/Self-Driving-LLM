from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Quaternion
import numpy as np
import json
import os
from tqdm import tqdm
import multiprocessing

# Load dataset
#nusc = NuScenes(version='v1.0-trainval', dataroot='/data/sets/nuscenes', verbose=True)

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
    heading = ego_rotation.yaw_pitch_roll[0]  # Extract yaw as heading

    # Get current object positions
    current_objects = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        obj_translation = np.array(ann['translation'])
        relative_position = ego_rotation.inverse.rotate(obj_translation - ego_translation)
        current_objects.append({'category': ann['category_name'], 'position': relative_position.tolist()})

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
        future_ego_positions[future_time] = relative_future_translation.tolist()

        # Get future object positions
        future_objects[future_time] = []
        for ann_token in future_sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            obj_translation = np.array(ann['translation'])
            relative_position = ego_rotation.inverse.rotate(obj_translation - future_translation)
            future_objects[future_time].append({'category': ann['category_name'], 'position': relative_position.tolist()})

    return {
        "instruction": "Given the current speed, heading, and object positions, predict the future ego and object positions relative to the current ego position.",
        "input": {"speed": velocity, "heading": heading, "current_objects": current_objects},
        "output": {"future_ego_positions": future_ego_positions, "future_objects": future_objects}
    }

def process_and_save(sample_token):
    """Processes a single sample and writes it directly to file in JSONL format."""
    result = get_local_positions(sample_token)

    # Write each sample separately to avoid memory issues
    with open("unsloth_data.json", "a") as f:
        f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    # Number of parallel processes (limit to 4-6 to avoid memory overload)
    num_workers = min(6, os.cpu_count() // 2)

    # Open file in write mode to clear old data
    with open("unsloth_data.json", "w") as f:
        pass  # Just to clear old content

    # Use multiprocessing to parallelize but with controlled memory usage
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_and_save, [s['token'] for s in nusc.sample]), total=len(nusc.sample), desc="Processing samples"))

    print("Processing complete! Data saved in 'unsloth_data.json'.")
