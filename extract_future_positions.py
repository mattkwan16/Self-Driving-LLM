from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Quaternion
import numpy as np
import json
import os
from tqdm import tqdm
import multiprocessing

# Load dataset
# nusc = NuScenes(version='v1.0-trainval', dataroot='/data/sets/nuscenes', verbose=True)

MAX_DISTANCE = 50
FUTURE_TIMES = [3, 6, 10]

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

def direction_from_angle(angle):
    angle = angle % 360
    clock_pos = round(((angle + 15) % 360) / 30)
    if clock_pos == 0:
        clock_pos = 12
    return f"{clock_pos} o'clock"

def get_local_positions(sample_token, future_times=[1, 3, 5], max_distance=None):
    """Retrieve ego and object positions relative to the ego vehicle at future timestamps."""
    count_omits = 0
    omitted_distances = []
    omitted_per_output = 0
    number_of_objects = 0
    sample = nusc.get('sample', sample_token)
    current_time = sample['timestamp'] / 1e6  # Convert to seconds

    # Get current ego pose
    ego_pose_data = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])
    ego_translation = np.array(ego_pose_data['translation'])
    ego_rotation = Quaternion(ego_pose_data['rotation'])

   # Get previous sample for velocity calculation
    prev_sample = sample
    while prev_sample['prev']:
        prev_sample = nusc.get('sample', prev_sample['prev'])
        break  # Break after getting the immediate previous sample

    prev_ego_pose_data = nusc.get('ego_pose', nusc.get('sample_data', prev_sample['data']['LIDAR_TOP'])['ego_pose_token'])
    prev_translation = np.array(prev_ego_pose_data['translation'])
    time_diff = current_time - (prev_sample['timestamp'] / 1e6) 
    velocity = np.linalg.norm(ego_translation - prev_translation) / time_diff if time_diff > 0 else 0  # Handle case where time_diff is 0

    heading = np.degrees(ego_rotation.yaw_pitch_roll[0])

    current_objects = []
    number_of_objects += len(sample['anns'])
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        obj_translation = np.array(ann['translation'])
        relative_position = ego_rotation.inverse.rotate(obj_translation - ego_translation)
        polar_position = cartesian_to_polar(relative_position)

        if max_distance is None or polar_position[1] <= max_distance:
            current_objects.append(f"There is a {ann['category_name']} {round(polar_position[1])}m away at your {direction_from_angle(polar_position[0])}.")
        else:
            count_omits += 1
            omitted_distances.append(polar_position[1])
            omitted_per_output += 1

    future_ego_positions, future_objects = {}, {}
    movement_instructions = []
    future_sample = sample
    for future_time in future_times:
        future_time_target = current_time + future_time

        while future_sample['next']:
            future_sample = nusc.get('sample', future_sample['next'])
            if future_sample['timestamp'] / 1e6 >= future_time_target:
                break

        ego_pose_data = nusc.get('ego_pose', nusc.get('sample_data', future_sample['data']['LIDAR_TOP'])['ego_pose_token'])
        future_translation = np.array(ego_pose_data['translation'])
        relative_future_translation = ego_rotation.inverse.rotate(future_translation - ego_translation)
        polar_future = cartesian_to_polar(relative_future_translation)
        future_ego_positions[future_time] = polar_future

        direction = direction_from_angle(polar_future[0])
        movement_instructions.append(f"In {future_time} second(s), arrive {round(polar_future[1], 1)}m to your {direction}.")

        future_objects[future_time] = []
        number_of_objects += len(future_sample['anns'])
        for ann_token in future_sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            obj_translation = np.array(ann['translation'])
            relative_position = ego_rotation.inverse.rotate(obj_translation - future_translation)
            polar_position = cartesian_to_polar(relative_position)

            if max_distance is None or polar_position[1] <= max_distance:
                future_objects[future_time].append({
                    'category': ann['category_name'],
                    'position': polar_position
                })
            else:
                count_omits += 1
                omitted_distances.append(polar_position[1])
                omitted_per_output += 1

    speed_text = f"Your speed is {round_floats(velocity)} m/s."
    heading_text = f"Your heading is {round_floats(heading)} degrees."

    return {
        "output_json": {
            "instruction": "Given the surrounding object descriptions, predict the ego vehicle's movement instructions.",
            "input": f"{speed_text} {heading_text} " + ' '.join(current_objects),
            "output": movement_instructions
        },
        "current_objects": current_objects,
        "future_ego_positions": future_ego_positions,
        "future_objects": future_objects,
        "omitted_distances": omitted_distances,
        "omitted_per_output": omitted_per_output,
        "number_of_objects": number_of_objects
    }

def process_sample(sample_token):
    return get_local_positions(sample_token, max_distance=MAX_DISTANCE, future_times=FUTURE_TIMES)

def process_and_save(sample_tokens):
    num_workers = 16

    results = []
    omitted_distances_total = []
    omitted_per_output_total = 0
    count_omits_total = 0
    total_objects_processed = 0  # Track total objects processed

    with multiprocessing.Pool(num_workers) as pool:
        for result in tqdm(pool.imap(process_sample, sample_tokens), total=len(sample_tokens), desc="Processing Samples"):
            results.append(result['output_json'])
            omitted_distances_total.extend(result['omitted_distances'])
            omitted_per_output_total += result['omitted_per_output']
            count_omits_total += len(result['omitted_distances'])
            total_objects_processed += result['number_of_objects']  # Add current objects processed

    # Total samples processed
    total_samples = len(sample_tokens)

    # Compute final statistics
    avg_omitted_distance = round(np.mean(omitted_distances_total), 3) if omitted_distances_total else 0
    avg_omitted_per_output = round(omitted_per_output_total / total_samples, 3) if total_samples else 0
    omitted_percentage = round((count_omits_total / total_objects_processed) * 100, 2) if total_samples else 0

    print(f"Total samples processed: {total_samples}")
    print(f"Total objects processed: {total_objects_processed}")  # Output total objects processed
    print(f"Total omitted objects: {count_omits_total}")
    print(f"Average omitted distance: {avg_omitted_distance}m")
    print(f"Max distance: {MAX_DISTANCE}m")
    print(f"Average omitted per output: {avg_omitted_per_output} objects")
    print(f"Omitted objects percentage: {omitted_percentage}%")

    # Save results as a valid JSON array
    with open("unsloth_data.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    # Get the list of sample tokens (assuming `nusc` is initialized)
    sample_tokens = [s['token'] for s in nusc.sample]

    # Run the processing
    process_and_save(sample_tokens)

    print("Processing complete! Data saved in 'unsloth_data.json'.")