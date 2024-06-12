import json
import os

def combine_json_files(root_dir, output_file):
    combined_data = []

    # Traverse through each sub-directory in the root_dir
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                # Create full file path and open each JSON file
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)  # Load data which is a single JSON object
                    combined_data.append(data)  # Append the whole object to the combined list

    # Write all combined data to a single output file
    with open(output_file, 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)

if __name__ == '__main__':
    base_dir = 'MATH'  # Path to the MATH directory
    test_dir = os.path.join(base_dir, 'test')
    train_dir = os.path.join(base_dir, 'train')
    
    combine_json_files(test_dir, 'combined_test.json')
    combine_json_files(train_dir, 'combined_train.json')

