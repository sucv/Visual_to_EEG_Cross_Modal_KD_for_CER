from utils.video import change_video_fps, combine_annotated_clips, OpenFaceController

from utils.utils import ensure_dir, get_filename_from_a_folder_given_extension, save_to_pickle

import os

from operator import itemgetter
from tqdm import tqdm

import pandas as pd
import numpy as np
from PIL import Image


class GenericVideoPreprocessing(object):
    def __init__(self, config):

        self.config = config
        self.dataset_info = self.init_dataset_info()

        if "extract_continuous_label" in config and config['extract_continuous_label']:
            self.extract_continuous_label = config['extract_continuous_label']

        if "change_video_fps" in config and config['change_video_fps']:
            self.change_video_fps = config['change_video_fps']
            self.fps_changed_video_folder = config['fps_changed_video_folder']

        if "trim_video" in config and config['trim_video']:
            self.trim_video = config['trim_video']
            self.trimmed_video_folder = config['trimmed_video_folder']

        if "crop_align_face" in config and config['crop_align_face']:
            self.crop_align_face = config['crop_align_face']
            self.cropped_aligned_folder = config['cropped_aligned_folder']

        if "extract_facial_landmark" in config and config['extract_facial_landmark']:
            self.extract_facial_landmark = config['extract_facial_landmark']
            self.facial_landmark_folder = config['facial_landmark_folder']

        if "extract_eeg" in config and config['extract_eeg']:
            from utils.eeg import GenericEegController
            self.extract_eeg = config['extract_eeg']
            self.eeg_folder = config['eeg_folder']

        self.per_trial_info = {}

    def get_output_root_directory(self):
        return self.config['output_root_directory']

    def prepare_data(self):

        for idx in tqdm(self.per_trial_info.keys(), total=len(self.per_trial_info.keys())):

            self.per_trial_info[idx]['processing_record'] = {}
            get_output_filename_kwargs = {}
            get_output_filename_kwargs['subject_no'] = self.per_trial_info[idx]['subject_no']
            get_output_filename_kwargs['trial_no'] = self.per_trial_info[idx]['trial_no']
            get_output_filename_kwargs['trial_name'] = self.per_trial_info[idx]['trial']
            output_filename = self.get_output_filename(**get_output_filename_kwargs)
            npy_folder = os.path.join(self.config['output_root_directory'], self.config['npy_folder'], output_filename)
            ensure_dir(npy_folder)

            self.per_trial_info[idx]['processing_record']['trial'] = output_filename

            self.per_trial_info[idx]['video_npy_path'] = os.path.join(npy_folder, "video.npy")

            # Load the continuous labels
            if hasattr(self, 'extract_continuous_label'):
                self.extract_continuous_label_fn(idx, npy_folder)

            ### VIDEO PREPROCESSING
            # Pick only the annotated frames from a video.
            if hasattr(self, 'trim_video'):
                self.trim_video_fn(idx, output_filename)

            # Change video fps to 64fps for this trial.
            if hasattr(self, 'change_video_fps'):
                self.change_video_fps_fn(idx, output_filename)

            # Extract facial landmark, warp, crop, and save each frame.
            if hasattr(self, 'crop_align_face'):
                self.crop_align_face_fn(idx, output_filename, npy_folder)

            if hasattr(self, "extract_facial_landmark"):
                self.extract_facial_landmark_fn(idx, output_filename, npy_folder)

            # EEG processing
            if hasattr(self, 'extract_eeg'):
                self.extract_eeg_fn(idx, output_filename, npy_folder)

        path = os.path.join(self.config['output_root_directory'], 'processing_records.pkl')
        ensure_dir(path)
        save_to_pickle(path, self.per_trial_info)

        self.generate_dataset_info()

    def extract_eeg_fn(self, idx, output_filename, npy_folder):

        if self.per_trial_info[idx]['has_eeg']:
            not_done = 0
            for feature in self.config['eeg_config']['features']:
                filename = os.path.join(npy_folder, feature + ".npy")
                if not os.path.isfile(filename):
                    not_done = 1

            if "eeg_processed_path" in self.per_trial_info[idx]:
                output_path = os.path.join(self.config['output_root_directory'],
                                           self.per_trial_info[idx]['eeg_processed_path'][-2:])
            else:

                # input_path = self.per_trial_info[idx]['eeg_path']
                input_path = os.path.join(self.config['root_directory'], *self.per_trial_info[idx]['eeg_path'][-3:])
                output_path = os.path.join(self.config['output_root_directory'], self.config['eeg_folder'],
                                           output_filename)

                if not_done:
                    from utils.eeg import GenericEegController
                    eeg_handler = GenericEegController(input_path, config=self.config['eeg_config'])

            self.per_trial_info[idx]['processing_record']['eeg_processed_path'] = output_path
            self.per_trial_info[idx]['eeg_processed_path'] = output_path.split(os.sep)

            if self.config['save_npy']:
                for feature in self.config['eeg_config']['features']:

                    filename = os.path.join(npy_folder, feature + ".npy")
                    self.per_trial_info[idx]['eeg_' + feature + '_npy_path'] = filename
                    if not os.path.isfile(filename):
                        # Save video npy
                        feature_np = eeg_handler.extracted_data[feature]
                        np.save(filename, feature_np)

    def generate_dataset_info(self):

        for idx, record in self.per_trial_info.items():
            self.dataset_info['trial'].append(record['processing_record']['trial'])
            self.dataset_info['trial_no'].append(record['trial_no'])
            self.dataset_info['subject_no'].append(record['subject_no'])
            self.dataset_info['length'].append(len(self.per_trial_info[idx]['continuous_label']))
            self.dataset_info['partition'].append(record['partition'])

        self.dataset_info['multiplier'] = self.config['multiplier']
        self.dataset_info['data_folder'] = self.config['npy_folder']

        path = os.path.join(self.config['output_root_directory'], 'dataset_info.pkl')
        save_to_pickle(path, self.dataset_info)

    def trim_video_fn(self, idx, output_filename):

        input_path = os.path.join(self.config['root_directory'], *self.per_trial_info[idx]['video_path'][-3:])
        output_path = os.path.join(self.config['output_root_directory'], self.config['trimmed_video_folder'],
                                   output_filename + ".mp4")

        ensure_dir(output_path)
        trim_range = self.per_trial_info[idx]['video_trim_range']
        combine_annotated_clips(input_path, output_path, trim_range, direct_copy=False, visualize=False)

        self.per_trial_info[idx]['processing_record']['trimmed_video_path'] = output_path.split(os.sep)

        self.per_trial_info[idx]['video_path'] = output_path.split(os.sep)

    def change_video_fps_fn(self, idx, output_filename):

        input_path = os.path.join(self.config['output_root_directory'], *self.per_trial_info[idx]['video_path'][-2:])
        output_path = os.path.join(self.config['output_root_directory'], self.config['fps_changed_video_folder'],
                                   output_filename + "." + self.per_trial_info[idx]['extension'])

        ensure_dir(output_path)
        change_video_fps(input_path, output_path, self.per_trial_info[idx]['target_fps'])

        self.per_trial_info[idx]['processing_record']['fps_video_path'] = output_path.split(os.sep)

        self.per_trial_info[idx]['video_path'] = output_path.split(os.sep)

    def crop_align_face_fn(self, idx, output_filename, npy_folder):

        openface_output_directory = os.path.join(self.config['output_root_directory'],
                                                 self.config['cropped_aligned_folder'])
        openface = OpenFaceController(openface_config=self.config['openface'],
                                      output_directory=openface_output_directory)

        output_path = os.path.join(openface_output_directory, output_filename)
        ensure_dir(openface_output_directory)
        input_path = os.path.join(self.config['output_root_directory'], *self.per_trial_info[idx]['video_path'][-2:])
        openface.process_video(input_filename=input_path,
                                             output_filename=output_filename)


        self.per_trial_info[idx]['processing_record']['cropped_aligned_path'] = output_path.split(os.sep)
        if self.config['save_npy']:
            filename = os.path.join(npy_folder, "video.npy")
            self.per_trial_info[idx]['video_npy_path'] = filename
            if not os.path.isfile(filename):
                # Save video npy
                annotated_index = self.per_trial_info[idx]['video_annotated_index']
                video_matrix = self.compact_facial_image(output_path,
                                                         annotated_index=annotated_index,
                                                         extension="jpg")
                np.save(filename, video_matrix)

    def extract_facial_landmark_fn(self, idx, output_filename, npy_folder):

        output_path = os.path.join(self.config['output_root_directory'], self.config['facial_landmark_folder'],
                                   output_filename + ".csv")

        self.per_trial_info[idx]['processing_record']['facial_landmark_path'] = output_path.split(os.sep)

        if self.config['save_npy']:

            filename = os.path.join(npy_folder, "landmark.npy")
            if not os.path.isfile(filename):
                # Save facial landmarks
                start_col, end_col = 5, 141
                feature = "facial_landmark"
                annotated_index = self.per_trial_info[idx]['video_annotated_index']
                landmark = self.compact_audio_feature(output_path, annotated_index, start_col, end_col, feature)

                np.save(filename, landmark)

    def extract_action_unit_fn(self, idx, output_filename, npy_folder):

        output_path = os.path.join(self.config['output_root_directory'], self.config['action_unit_folder'],
                                   output_filename + ".csv")

        self.per_trial_info[idx]['processing_record']['action_unit_path'] = output_path.split(os.sep)
        if self.config['save_npy']:
            filename = os.path.join(npy_folder, "action_unit.npy")
            if not os.path.isfile(filename):
                # Save facial landmarks
                start_col = 141
                end_col = 158
                feature = "action_unit"
                annotated_index = self.per_trial_info[idx]['video_annotated_index']
                action_unit = self.compact_audio_feature(output_path, annotated_index, start_col, end_col, feature)

                np.save(filename, action_unit)

    def extract_continuous_label_fn(self, idx, npy_folder):

        if not 'continuous_label' in self.per_trial_info[idx]:
            raw_continuous_label = self.load_continuous_label(self.per_trial_info[idx]['continuous_label_path'])

            self.per_trial_info[idx]['continuous_label'] = raw_continuous_label[self.per_trial_info[idx]['annotated_index']]

        if self.config['save_npy']:
            filename = os.path.join(npy_folder, "continuous_label.npy")
            if not os.path.isfile(filename):
                ensure_dir(filename)
                np.save(filename, self.per_trial_info[idx]['continuous_label'])

    def extract_class_label_fn(self, record):
        pass

    def load_continuous_label(self, path, **kwargs):
        raise NotImplementedError

    def compact_audio_feature(self, path, annotated_index, start_col=0, end_col=0, feature=""):

        length = max(annotated_index)

        sep = ";"
        if feature == "facial_landmark" or feature == "action_unit":
            sep = ","

        feature_matrix = pd.read_csv(path, sep=sep, usecols=range(start_col, end_col)).values

        # If the continuous label is longer than the video, then repetitively pad (edge padding) the last element.
        length_difference = length - len(feature_matrix) + 1

        if length_difference > 0:
            feature_matrix = np.vstack(
                (feature_matrix, np.repeat(feature_matrix[-1, :][None, :], length_difference, axis=0)))

        feature_matrix = feature_matrix[annotated_index]

        return feature_matrix

    def compact_facial_image(self, path, annotated_index, extension="jpg"):

        length = len(annotated_index)

        facial_image_list = get_filename_from_a_folder_given_extension(path + "_aligned", extension)

        # If the continuous label is longer than the video, then repetitively pad (edge padding) the last element.
        length_difference = length - len(facial_image_list)
        if length_difference:
            [facial_image_list.extend([facial_image_list[-1]]) for _ in range(length_difference)]

        facial_image_list = list(itemgetter(*annotated_index)(facial_image_list))

        frame_matrix = np.zeros((
            length, self.config['video_size'], self.config['video_size'], 3), dtype=np.uint8)

        for j, frame in enumerate(facial_image_list):
            frame_matrix[j] = Image.open(frame)

        return frame_matrix

    def process_continuous_label(self, continuous_label):
        return list(range(len(continuous_label)))

    def generate_iterator(self):
        return NotImplementedError

    def generate_per_trial_info_dict(self):
        raise NotImplementedError

    def get_video_trim_range(self):
        trim_range = []
        return trim_range

    def get_annotated_index(self, annotated_index):
        return annotated_index

    @staticmethod
    def get_output_filename(**kwargs):

        output_filename = "P{}-T{}".format(kwargs['subject_no'], kwargs['trial_no'])
        return output_filename

    @staticmethod
    def init_dataset_info():
        dataset_info = {
            "trial": [],
            "subject_no": [],
            "trial_no": [],
            "length": [],
            "partition": [],
        }
        return dataset_info



