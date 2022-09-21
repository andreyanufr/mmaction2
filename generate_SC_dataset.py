import os
import random
import shutil
import argparse
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('src_dir', type=str, help='path to directory with new dataset')
    parser.add_argument('src_mapping_name', type=str,
                    help='path to file with source dataset mapping (for example label_map_k400.txt)')
    parser.add_argument('experiment_name', type=str,
                        help='name of experiment. This name will be used for new dataset path')
    parser.add_argument('--n_classes', type=int, default=3,
                        help='number of classes for SC scenario')
    parser.add_argument('--n_samples', type=int, default=12,
                        help='number of samples classes for SC scenario (50% train, 25 % val and test)')
    parser.add_argument('--n_min_frames', type=int, default=60,
                        help='number of minimum frames per one sample (threshold)')
    parser.add_argument('--n_train_ratio', type=float, default=0.5,
                    help='ratio of training samples')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    return args


def load_raw_frames_dataset(file_name, used_data=set(), min_frames=60):
    print("load_raw_frames_dataset from file: ", file_name)
    res = []
    class_partition = {}

    with open(file_name, 'r') as f:
        lines = f.read().splitlines()
        for idx, line in enumerate(lines):
            vals = line.rstrip().split(' ')  # vals[0] = doing_laundry/WxDh_2AEBhc_000212_000222

            if 'jester' in file_name:
                key = vals[0]
            else:
                key = vals[0].split('/')[1][:-14]

            if key in used_data:  # avoid dataset intersection
                continue
            if int(vals[1]) < min_frames:
                continue
            res.append([vals[0], int(vals[1]), int(vals[2])])  # video name, number of frames, label_id
            
            if res[-1][-1] not in class_partition:
                class_partition[res[-1][-1]] = [len(res) - 1]
            else:
                class_partition[res[-1][-1]].append(len(res) - 1)

    return res, class_partition


def load_annotation_dataset(file_name):
    res = []
    counter = 0
    with open(file_name, 'r') as f:
        for line in f:
            counter += 1
            if counter == 1:
                continue
            vals = line.rstrip().split(',')
            res.append(vals[1])

    return set(res)


def load_mapping(file_name):
    label_id = 0
    res = {}
    with open(file_name, 'r') as f:
        for line in f:
            line = line.rstrip()
            res[line] = label_id
            label_id += 1
    return res


def get_name(pattern):
    return glob(pattern)[0]


def create_subset(dst_dir, src_dir, data_idxs, data, split, mapping_to_new_class, new_mapping_dict):
    dst_video_dir = os.path.join(dst_dir, 'rawframes_' + split)
    os.makedirs(dst_video_dir, exist_ok=True)
    src_frames_dir = os.path.join(src_dir, 'rawframes')

    print(os.path.join(dst_dir, split + '_list_rawframes.txt'))
    with open(os.path.join(dst_dir, split + '_list_rawframes.txt'), 'w') as f:
        for idx in data_idxs:
            cur_data = data[idx]
            cur_data[-1] = mapping_to_new_class[cur_data[-1]]
            f.write(" ".join(str(val) for val in cur_data) + "\n")
    with open(os.path.join(dst_dir, 'label_map.txt'), 'w') as f:
        for value in new_mapping_dict.values():
            f.write(value+"\n")

    for idx in data_idxs:
        cur_data = data[idx]
        shutil.copytree(os.path.join(src_frames_dir, cur_data[0]),
                os.path.join(dst_video_dir, cur_data[0]))



def create_sc_subset(src_dir, experiment_name, src_mapping_name,
                     num_classes=3, num_samples=12, min_frames=60,
                     train_ratio=0.5):
    src_data, src_data_class_mapping = load_raw_frames_dataset(
        get_name(os.path.join(src_dir, '*val*rawframes*.txt')), min_frames=min_frames)

    src_mapping = load_mapping(src_mapping_name)

    experiment_dir = os.path.join(src_dir, experiment_name)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    processed = set()
    res = 0

    dst_keys = list(src_mapping.keys())

    num_train = int(num_samples * train_ratio)
    num_val = (num_samples - num_train) // 2
    num_test = num_samples - num_train - num_val

    assert num_test > 0

    random.shuffle(dst_keys)
    train_idxs = []
    val_idxs = []
    test_idxs = []

    new_class_mapping = {}
    while res < num_classes:
        idx = random.randint(0, len(dst_keys) - 1)
        if dst_keys[idx] in processed:
            continue
        key_name = dst_keys[idx]
        processed.add(key_name)

        data_idxs = src_data_class_mapping[src_mapping[key_name]]

        if len(data_idxs) < num_samples:
            continue

        print("Key : ", key_name)

        random.shuffle(data_idxs)
        new_data_idxs = data_idxs[:num_samples]

        new_class_mapping[src_mapping[key_name]] = res

        train_idxs.extend(new_data_idxs[:num_train])
        val_idxs.extend(new_data_idxs[num_train:num_train+num_val])
        test_idxs.extend(new_data_idxs[num_train+num_val:num_train+num_val+num_test])

        res += 1
    all_classes = [i for i in src_mapping.keys()]
    new_mapping_dict = {idx:all_classes[key] for idx, key in enumerate(new_class_mapping.keys())}
    
    create_subset(experiment_dir, src_dir, train_idxs, src_data, 'train', new_class_mapping, new_mapping_dict)
    create_subset(experiment_dir, src_dir, val_idxs, src_data, 'val', new_class_mapping, new_mapping_dict)
    create_subset(experiment_dir, src_dir, test_idxs, src_data, 'test', new_class_mapping, new_mapping_dict)

    return res




if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    res = create_sc_subset(args.src_dir,
                     args.experiment_name,
                     args.src_mapping_name,
                     num_classes=args.n_classes,
                     num_samples=args.n_samples,
                     min_frames=args.n_min_frames,
                     train_ratio=args.n_train_ratio)

    print(res)
