# -*- coding: utf-8 -*-
import numpy as np
import json
import os
import argparse

parser = argparse.ArgumentParser(description='Prepare Speech Commands')
parser.add_argument('--dataset_path', type=str, required=True, help='Path where the dataset is unzipped (e.g. /scratch/speech_commands_v0.02)')
args = parser.parse_args()

# Use the provided path
base_path = args.dataset_path
if not base_path.endswith('/'):
    base_path += '/'

print(f'Processing data from: {base_path}')

# NOTE: Remove the download logic here because we will handle it in the bash script 
# to ensure it comes from AFS -> Scratch.

# generate training list
if os.path.exists(os.path.join(base_path, 'train_list.txt')) == False:
    with open(os.path.join(base_path, 'validation_list.txt'), 'r') as f:
        val_list = f.readlines()

    with open(os.path.join(base_path, 'testing_list.txt'), 'r') as f:
        test_list = f.readlines()

    val_test_list = list(set(test_list+val_list))

    def get_immediate_subdirectories(a_dir):
        return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]
    def get_immediate_files(a_dir):
        return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

    all_cmds = get_immediate_subdirectories(base_path)
    all_list = []
    for cmd in all_cmds:
        if cmd != '_background_noise_':
            cmd_samples = get_immediate_files(base_path+'/'+cmd)
            for sample in cmd_samples:
                all_list.append(cmd + '/' + sample+'\n')

    training_list = [x for x in all_list if x not in val_test_list]

    with open(os.path.join(base_path, 'train_list.txt'), 'w') as f:
        f.writelines(training_list)

# Load labels (Keep this pointing to the local repo file)
label_set = np.loadtxt('./data/speechcommands_class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

# generate json files
if os.path.exists('./data/datafiles') == False:
    os.mkdir('./data/datafiles')
    
# Always regenerate JSONs to ensure they point to the correct Scratch path
for split in ['testing', 'validation', 'train']:
    wav_list = []
    # Read the list file from the Scratch directory
    with open(base_path + split + '_list.txt', 'r') as f:
        filelist = f.readlines()
    for file in filelist:
        cur_label = label_map[file.split('/')[0]]
        # 2. Critical Change: Use the absolute path provided by args.dataset_path
        cur_path = os.path.join(base_path, file.strip())
        cur_dict = {"wav": cur_path, "labels": '/m/spcmd'+cur_label.zfill(2)}
        wav_list.append(cur_dict)
    if split == 'train':
        with open('./data/datafiles/speechcommand_train_data.json', 'w') as f:
            json.dump({'data': wav_list}, f, indent=1)
    if split == 'testing':
        with open('./data/datafiles/speechcommand_eval_data.json', 'w') as f:
            json.dump({'data': wav_list}, f, indent=1)
    if split == 'validation':
        with open('./data/datafiles/speechcommand_valid_data.json', 'w') as f:
            json.dump({'data': wav_list}, f, indent=1)
    print(split + ' data processing finished, total {:d} samples'.format(len(wav_list)))

print('Speechcommands dataset processing finished.')