from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Quaternion
import numpy as np
import json
import os
from tqdm import tqdm
from joblib import Parallel, delayed

# Load dataset
nusc = NuScenes(version='v1.0-trainval', dataroot='/data/sets/nuscenes', verbose=True)

# Precompute timestamp-to-sample mapping for fast lookup
timestamp_to_sample = {s['timestamp']: s for s in nusc.sample}

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
    for future_time in future_times:
        future_time_target = current_time + future_time
        future_sample = min(
            (s for t, s in timestamp_to_sample.items() if t >= future_time_target * 1e6),
            key=lambda s: s['timestamp'],
            default=None
        )
        if not future_sample:
            continue

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

    return velocity, heading, current_objects, future_ego_positions, future_objects

def process_sample(sample):
    """Processes a single sample and writes it to file."""
    sample_token = sample['token']
    velocity, heading, current_positions, future_ego_positions, future_positions = get_local_positions(sample_token)

    entry = {
        "instruction": "Given the current speed, heading, and object positions, predict the future ego and object positions relative to the current ego position.",
        "input": {"speed": velocity, "heading": heading, "current_objects": current_positions},
        "output": {"future_ego_positions": future_ego_positions, "future_objects": future_positions}
    }

    # Stream-write the result to avoid memory issues
    with open("unsloth_data.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

# Process dataset in parallel, writing incrementally
if __name__ == "__main__":
    num_workers = min(8, os.cpu_count())  # Use up to 8 cores
    Parallel(n_jobs=num_workers)(
        delayed(process_sample)(sample) for sample in tqdm(nusc.sample, desc="Processing samples", unit="sample")
    )

print("Processing complete! Data saved in 'unsloth_data.jsonl'.")
