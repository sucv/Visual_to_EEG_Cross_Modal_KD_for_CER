from utils.transforms3D import *
from utils.utils import load_npy

from torch.utils.data import Dataset
from torchvision.transforms import transforms

import os
from operator import itemgetter


class MahnobDataset(Dataset):
    def __init__(self, data_list, continuous_label_dim, modality, multiplier, feature_dimension, window_length, mode,
                 mean_std=None,
                 time_delay=0, load_whole_trial=0, length_modifier=1, knowledge_path=''):

        self.data_list = data_list
        self.continuous_label_dim = continuous_label_dim
        self.mean_std = mean_std
        self.time_delay = time_delay
        self.modality = modality
        self.multiplier = multiplier
        self.feature_dimension = feature_dimension
        self.load_whole_trial = load_whole_trial
        self.length_modifier = length_modifier
        self.window_length = window_length
        self.mode = mode
        self.transform_dict = {}
        self.get_3D_transforms()

        self.knowledge_path = knowledge_path

    def __getitem__(self, index):
        path, trial, length, index = self.data_list[index]
        length *= self.length_modifier

        examples = {}

        for feature in self.modality:
            examples[feature] = self.get_example(path, length, index, feature)

        if self.knowledge_path != "" and self.mode != "test":
            knowledge_path = os.path.join(self.knowledge_path, trial)
            examples['visual_knowledge'] = self.get_example(knowledge_path, length, index, "visual_knowledge")

        if len(index) < self.window_length:
            index = np.arange(self.window_length)

        return examples, str(trial), length, index

    def __len__(self):
        return len(self.data_list)

    def get_example(self, path, length, index, feature):

        x = random.randint(0, self.multiplier[feature] - 1)
        random_index = index * self.multiplier[feature] + x

        # Probably, a trial may be shorter than the window, so the zero padding is employed.
        if length < self.window_length:
            shape = (self.window_length,) + self.feature_dimension[feature]
            dtype = np.float32
            if feature == "video":
                dtype = np.int8
            example = np.zeros(shape=shape, dtype=dtype)
            example[index] = self.load_data(path, random_index, feature)
        else:
            example = self.load_data(path, random_index, feature)

        # Sometimes we may want to shift the label, so that
        # the ith label point  corresponds to the (i - time_delay)-th data point.
        if "continuous_label" in feature and self.time_delay != 0:
            example = np.concatenate(
                (example[self.time_delay:, :],
                 np.repeat(example[-1, :][np.newaxis], repeats=self.time_delay, axis=0)), axis=0)

        if "continuous_label" not in feature and "visual_knowledge" not in feature:
            example = self.transform_dict[feature](np.asarray(example, dtype=np.float32))

        return example

    def get_3D_transforms(self):
        normalize = GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if "video" in self.modality:
            if self.mode == 'train':
                self.transform_dict['video'] = transforms.Compose([
                    GroupNumpyToPILImage(0),
                    GroupRandomCrop(48, 40),
                    GroupRandomHorizontalFlip(),
                    Stack(),
                    ToTorchFormatTensor(),
                    normalize
                ])
            else:
                self.transform_dict['video'] = transforms.Compose([
                    GroupNumpyToPILImage(0),
                    GroupCenterCrop(40),
                    Stack(),
                    ToTorchFormatTensor(),
                    normalize
                ])

        for feature in self.modality:
            if "continuous_label" not in feature and "video" not in feature:
                self.transform_dict[feature] = self.get_feature_transform(feature)

    def get_feature_transform(self, feature):
        if "cnn" in feature or "eeg_bandpower" in feature or "vggface" in feature:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean_std[feature]['mean']],
                                     std=[self.mean_std[feature]['std']])
            ])
        return transform

    def load_data(self, path, indices, feature):
        filename = os.path.join(path, feature + ".npy")

        # For the test set, labels of zeros are generated as dummies.
        data = np.zeros(((len(indices),) + self.feature_dimension[feature]), dtype=np.float32)

        if os.path.isfile(filename):
            if self.load_whole_trial:
                data = np.load(filename, mmap_mode='c')
            else:
                data = np.load(filename, mmap_mode='c')[indices]

            if "continuous_label" in feature:
                data = self.processing_label(data)
        else:
            if "knowledge" in feature:
                raise ValueError(
                    "The knowledge file does not exist, please extract it before running the knowledge distillation stage!")
        return data

    def processing_label(self, label):
        label = label[:, self.continuous_label_dim]
        if label.ndim == 1:
            label = label[:, None]
        return label


class DataArranger(object):
    def __init__(self, dataset_info, dataset_path, debug, case, seed, load_whole_trial=1):
        self.case = case
        self.seed = seed

        self.dataset_info = dataset_info
        self.debug = debug
        self.trial_list = self.generate_raw_trial_list(dataset_path)
        self.partition_range = self.partition_range_fn()
        self.fold_to_partition = self.assign_fold_to_partition()

    def generate_partitioned_trial_list(self, window_length, hop_length, fold, windowing=True):

        partition_range = list(np.roll(self.partition_range, fold))
        partitioned_trial = {}

        trial_list = self.trial_list

        for partition, num_fold in self.fold_to_partition.items():
            partitioned_trial[partition] = []
            if partition == "validate":
                partitioned_trial['validate'] = validate_trial

            for i in range(num_fold):
                index = partition_range.pop(0)
                trial_of_this_fold = list(itemgetter(*index)(trial_list))

                if len(index) == 1:
                    trial_of_this_fold = [trial_of_this_fold]

                for path, trial, length in trial_of_this_fold:
                    partitioned_trial[partition].append([path, trial, length])

            # Split the total N-1 folds into training and validation sets (60% to 40%).

            if partition == "train":
                random.Random(self.seed).shuffle(partitioned_trial['train'])

                count = len(partitioned_trial['train'])

                num_train_trials = int(count * 0.8)

                train_idx = np.arange(num_train_trials)
                validate_idx = np.arange(num_train_trials, count)

                train_trial = list(itemgetter(*train_idx)(partitioned_trial["train"]))
                validate_trial = list(itemgetter(*validate_idx)(partitioned_trial["train"]))

                partitioned_trial['train'] = train_trial

        windowed_partitioned_trial = {key: [] for key in partitioned_trial}
        for partition, trials in partitioned_trial.items():
            for path, trial, length in trials:

                if windowing:
                    windowed_indices = self.windowing(np.arange(length), window_length=window_length,
                                                      hop_length=hop_length)
                else:
                    windowed_indices = self.windowing(np.arange(length), window_length=length,
                                                      hop_length=hop_length)

                for index in windowed_indices:
                    windowed_partitioned_trial[partition].append([path, trial, length, index])

        return windowed_partitioned_trial

    def calculate_mean_std(self, partitioned_trial):
        # Why not just call .mean() and .std() on the data to calculate the mean and std?
        # To save the time and RAM. When data have millions of points, its std would take forever to finish.
        # Therefore, the calculation is done manually and segment-wise.
        feature_list = self.get_feature_list()
        mean_std_dict = {partition: {feature: {'mean': None, 'std': None} for feature in feature_list} for partition in
                         partitioned_trial.keys()}

        # Calculate the mean
        for feature in feature_list:
            for partition, trial_of_a_partition in partitioned_trial.items():
                lengths = 0
                sums = 0
                for path, _, _, _ in trial_of_a_partition:
                    data = load_npy(path, feature)
                    data = data.flatten()
                    lengths += len(data)
                    sums += data.sum()
                mean_std_dict[partition][feature]['mean'] = sums / (lengths + 1e-10)

        # Then calculate the standard deviation.
        for feature in feature_list:
            for partition, trial_of_a_partition in partitioned_trial.items():
                lengths = 0
                x_minus_mean_square = 0
                mean = mean_std_dict[partition][feature]['mean']
                for path, _, _, _ in trial_of_a_partition:
                    data = load_npy(path, feature)
                    data = data.flatten()
                    lengths += len(data)
                    x_minus_mean_square += np.sum((data - mean) ** 2)
                x_minus_mean_square_divide_N_minus_1 = x_minus_mean_square / (lengths - 1)
                mean_std_dict[partition][feature]['std'] = np.sqrt(x_minus_mean_square_divide_N_minus_1)

        return mean_std_dict

    def generate_raw_trial_list(self, dataset_path):
        trial_path = os.path.join(dataset_path, self.dataset_info['data_folder'])
        train_list = []

        for trial in self.generate_iterator():
            idx = self.dataset_info['trial'].index(trial)
            trial = self.dataset_info['trial'][idx]
            path = os.path.join(trial_path, trial)
            length = self.dataset_info['length'][idx]
            train_list.append([path, trial, length])

        return train_list

    def generate_iterator(self):
        iterator = []

        for idx, trial in enumerate(self.dataset_info['trial']):
            if self.dataset_info['has_continuous_label'][idx]:
                iterator.append(trial)
        return iterator

    def partition_range_fn(self):

        if self.case == "trs":
            partition_range = [np.arange(a, a + 24) for a in np.arange(0, 239, 24)]

            # Make sure the last fold (for testing) has 24 trials, i.e., 10% of the whole data.
            partition_range[-1] = np.insert(partition_range[-1], obj=[0], values=partition_range[-2][-1])
            partition_range[-2] = np.delete(partition_range[-2], [-1])
            partition_range[-1] = np.delete(partition_range[-1], [-1])

        elif self.case == "loso":
            partition_range = [np.arange(0, 19), np.arange(19, 24), np.arange(24, 37),
                               np.arange(37, 46), np.arange(46, 59), np.arange(59, 68),
                               np.arange(68, 81), np.arange(81, 94), np.arange(94, 99),
                               np.arange(99, 105), np.arange(105, 118), np.arange(118, 120),
                               np.arange(120, 121), np.arange(121, 132), np.arange(132, 149),
                               np.arange(149, 159), np.arange(159, 173), np.arange(173, 188),
                               np.arange(188, 200), np.arange(200, 216), np.arange(216, 222),
                               np.arange(222, 226), np.arange(226, 236), np.arange(236, 239)]

        if self.debug == 1:
            if self.case == "loso":
                num_folds = 24
            elif self.case == "trs":
                num_folds = 10
            else:
                raise ValueError("Unknown case!")
            partition_range = [np.arange(a, a + 1) for a in range(num_folds)]

        return partition_range

    def assign_fold_to_partition(self):
        if self.case == "trs":
            fold_to_partition = {'train': 9, 'validate': 0, 'test': 1}
        elif self.case == "loso":
            fold_to_partition = {'train': 23, 'validate': 0, 'test': 1}
        else:
            raise ValueError("Unknown case!!")

        return fold_to_partition

    @staticmethod
    def get_feature_list():
        # No normalization is required.
        feature_list = ['landmark']
        return feature_list

    @staticmethod
    def windowing(x, window_length, hop_length):
        length = len(x)

        if length >= window_length:
            steps = (length - window_length) // hop_length + 1

            sampled_x = []
            for i in range(steps):
                start = i * hop_length
                end = start + window_length
                sampled_x.append(x[start:end])

            if sampled_x[-1][-1] < length - 1:
                sampled_x.append(x[-window_length:])
        else:
            sampled_x = [x]

        return sampled_x