from utils.utils import load_pickle, ensure_dir, CCCLoss, save_to_pickle

from utils.trainer import Trainer
from utils.parameter_control import ResnetParamControl
from models.main_model import my_temporal
from utils.dataset import DataArranger, MahnobDataset
from utils.checkpointer import Checkpointer

import os
import numpy as np
import random
import torch


class Experiment(object):
    def __init__(self, args):
        # Basic experiment settings.
        self.experiment_name = args.experiment_name
        self.dataset_path = args.dataset_path
        self.load_path = args.load_path
        self.save_path = args.save_path
        self.stamp = args.stamp
        self.seed = args.seed
        self.resume = args.resume
        self.debug = args.debug
        self.calc_mean_std = args.calc_mean_std

        self.high_performance_cluster = args.high_performance_cluster
        self.gpu = args.gpu
        self.cpu = args.cpu
        self.device = self.init_device()
        # If the code is to run on high-performance computer, which is usually not
        # available to specify gpu index and cpu threads, then set them to none.
        if self.high_performance_cluster:
            self.gpu = None
            self.cpu = None

        self.model_name = args.model_name
        self.cross_validation = args.cross_validation
        self.folds_to_run = args.folds_to_run
        if not self.cross_validation:
            self.folds_to_run = [0]

        self.scheduler = args.scheduler
        self.learning_rate = args.learning_rate
        self.min_learning_rate = args.min_learning_rate
        self.factor = args.factor
        self.patience = args.patience
        self.modality = args.modality
        self.calc_mean_std = args.calc_mean_std
        self.seed = args.seed

        self.window_length = args.window_length
        self.hop_length = args.hop_length
        self.batch_size = args.batch_size

        self.extract_feature = args.extract_feature
        self.time_delay = None
        self.dataset_info = None
        self.mean_std_dict = None
        self.data_arranger = None

        self.feature_dimension = None
        self.multiplier = None
        self.continuous_label_dim = None

        self.config = None

        if 'emotion' in args:
            self.emotion = args.emotion

        self.args = args
        self.case = args.case
        self.min_epoch = args.min_num_epochs

        self.early_stopping = args.early_stopping
        self.metrics = args.metrics

        # For tcn and lstm on regression tasks.
        self.backbone_state_dict = args.backbone_state_dict
        self.cnn1d_channels = args.cnn1d_channels
        self.cnn1d_kernel_size = args.cnn1d_kernel_size
        self.cnn1d_dropout = args.cnn1d_dropout
        self.lstm_embedding_dim = args.lstm_embedding_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.lstm_dropout = args.lstm_dropout

        self.use_kd_on_student = args.use_kd_on_student
        self.w = args.w

        # For convenience
        if self.case == "loso":
            self.num_folds = 24
            if self.folds_to_run[0] == "all":
                self.folds_to_run = np.arange(24)
        elif self.case == "trs":
            self.num_folds = 10
            if self.folds_to_run[0] == "all":
                self.folds_to_run = np.arange(10)
        else:
            raise ValueError("Unknown case!")

        self.release_count = 0
        self.gradual_release = 0
        self.milestone = []
        self.max_epoch = 15
        self.input_size = 192
        if "video" in self.modality:
            self.release_count = 3
            self.gradual_release = 1
            self.milestone = [0]
            self.max_epoch = 30
            self.input_size = 512

        self.save_plot = args.save_plot
        self.load_best_at_each_epoch = args.load_best_at_each_epoch
        self.time_delay = args.time_delay
        self.continuous_label_dim = 0

        if self.extract_feature:
            assert "video" in self.modality, "Feature extraction is supposed to be applied on video modality only!"
            assert self.use_kd_on_student == 0, "Feature extraction and knowledge distillation are not supposed to run simultaneously."
        if self.use_kd_on_student:
            assert "eeg_bandpower" in self.modality, "Knowledge distillation is supposed to be applied on eeg modality only!"
            assert self.extract_feature == 0, "Feature extraction and knowledge distillation are not supposed to run simultaneously."

        if self.debug:
            self.max_epoch = 1

    def prepare(self):
        self.config = self.get_config()

        self.feature_dimension = self.get_feature_dimension(self.config)
        self.multiplier = self.get_multiplier(self.config)

        self.dataset_info = load_pickle(os.path.join(self.dataset_path, "dataset_info.pkl"))
        self.data_arranger = self.init_data_arranger()
        if self.calc_mean_std:
            self.calc_mean_std_fn()
        self.mean_std_dict = load_pickle(os.path.join(self.dataset_path, "mean_std_info.pkl"))

    def run(self):
        criterion = CCCLoss()

        load_whole_trial = 0
        if self.extract_feature:
            load_whole_trial = 1

        for fold in iter(self.folds_to_run):

            save_path_str = os.path.join(self.save_path,
                                         self.experiment_name + "_" + self.model_name + "_" + self.stamp + "_" + self.case,
                                         "{}", str(
                    fold) + "_" + self.emotion)
            knowledge_path = ""
            save_path = save_path_str.format(self.modality[0])
            if self.use_kd_on_student:
                save_path = save_path_str.format("taught_eeg")
                knowledge_path = os.path.join(save_path_str.format("video"), "visual_feature")

            ensure_dir(save_path)

            feature_path = os.path.join(save_path, "visual_feature")
            ensure_dir(feature_path)

            checkpoint_filename = os.path.join(save_path, "checkpoint.pkl")

            model = self.init_model()

            dataloaders = self.init_dataloader(fold, knowledge_path=knowledge_path)
            dataloaders_feature_extraction = self.init_dataloader(fold, load_whole_trial=load_whole_trial,
                                                                  length_modifier=1)

            trainer_kwards = {'device': self.device, 'emotion': self.emotion, 'model_name': self.model_name,
                              'models': model, 'save_path': save_path, 'fold': fold,
                              'min_epoch': self.min_epoch, 'max_epoch': self.max_epoch,
                              'early_stopping': self.early_stopping, 'scheduler': self.scheduler,
                              'learning_rate': self.learning_rate, 'min_learning_rate': self.min_learning_rate,
                              'patience': self.patience, 'batch_size': self.batch_size, 'w': self.w,
                              'criterion': criterion, 'factor': self.factor, 'verbose': True,
                              'milestone': self.milestone, 'metrics': self.metrics, 'feature_path': feature_path,
                              'load_best_at_each_epoch': self.load_best_at_each_epoch, 'save_plot': self.save_plot,
                              'emotion_dim': self.continuous_label_dim}

            trainer = Trainer(**trainer_kwards)

            parameter_controller = ResnetParamControl(trainer, gradual_release=self.gradual_release,
                                                      release_count=self.release_count, backbone_mode="ir")

            checkpoint_controller = Checkpointer(checkpoint_filename, trainer, parameter_controller, resume=self.resume)

            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
            else:
                checkpoint_controller.init_csv_logger(self.args, self.config)

            if not trainer.fit_finished:
                trainer.fit(dataloaders, parameter_controller=parameter_controller,
                            checkpoint_controller=checkpoint_controller)

            if not trainer.fold_finished and 'test' in dataloaders:
                test_kwargs = {'dataloader_dict': dataloaders, 'epoch': None, 'partition': 'test',
                               'scaler': self.mean_std_dict[fold]}
                trainer.test(checkpoint_controller, predict_only=0, **test_kwargs)
                checkpoint_controller.save_checkpoint(trainer, parameter_controller, save_path)

            if self.extract_feature:
                extract_kwargs = {'dataloader_dict': dataloaders_feature_extraction, 'epoch': None,
                                  'partition': 'train'}
                trainer.extract(**extract_kwargs)

                extract_kwargs = {'dataloader_dict': dataloaders_feature_extraction, 'epoch': None,
                                  'partition': 'validate'}
                trainer.extract(**extract_kwargs)

        self.print_output(save_path)

    def init_model(self):
        self.init_randomness()
        visual_backbone_path = os.path.join(self.load_path, self.backbone_state_dict + ".pth")
        model = my_temporal(model_name=self.model_name,
                            num_inputs=self.input_size, visual_backbone_path=visual_backbone_path,
                            cnn1d_channels=self.cnn1d_channels, cnn1d_kernel_size=self.cnn1d_kernel_size,
                            cnn1d_dropout_rate=self.cnn1d_dropout, embedding_dim=self.lstm_embedding_dim,
                            hidden_dim=self.lstm_hidden_dim, lstm_dropout_rate=self.lstm_dropout,
                            modality=self.modality, output_dim=1)
        return model

    def init_dataloader(self, fold, load_whole_trial=0, length_modifier=1, knowledge_path=''):

        if load_whole_trial:
            windowing = 0
            batch_size = 1
            length_modifier = length_modifier
        else:
            windowing = 1
            batch_size = self.batch_size
            length_modifier = 1

        self.init_randomness()
        data_list = self.data_arranger.generate_partitioned_trial_list(window_length=self.window_length,
                                                                       hop_length=self.hop_length, fold=fold,
                                                                       windowing=windowing)

        datasets, dataloaders = {}, {}
        for mode, data in data_list.items():
            if len(data):

                # Cross-platform deterministic shuffling for the training set.
                if mode == "train":
                    random.shuffle(data_list[mode])

                datasets[mode] = self.init_dataset(data_list[mode], self.continuous_label_dim, mode, fold,
                                                   load_whole_trial=load_whole_trial, length_modifier=length_modifier,
                                                   knowledge_path=knowledge_path)

                dataloaders[mode] = torch.utils.data.DataLoader(
                    dataset=datasets[mode], batch_size=batch_size, shuffle=False)

        return dataloaders

    def init_data_arranger(self):
        arranger = DataArranger(self.dataset_info, self.dataset_path, self.debug, self.case, self.seed)
        return arranger

    def init_dataset(self, data, continuous_label_dim, mode, fold, load_whole_trial=0, length_modifier=1,
                     knowledge_path=''):

        dataset = MahnobDataset(data, continuous_label_dim, self.modality, self.multiplier,
                          self.feature_dimension, self.window_length,
                          mode, mean_std=self.mean_std_dict[fold][mode], time_delay=self.time_delay,
                          load_whole_trial=load_whole_trial, length_modifier=length_modifier,
                          knowledge_path=knowledge_path)
        return dataset

    def calc_mean_std_fn(self):
        path = self.get_mean_std_dict_path()

        mean_std_dict = {}
        for fold in range(self.num_folds):
            data_list = self.data_arranger.generate_partitioned_trial_list(window_length=self.window_length,
                                                                           hop_length=self.hop_length, fold=fold,
                                                                           windowing=False)
            mean_std_dict[fold] = self.data_arranger.calculate_mean_std(data_list)

        save_to_pickle(path, mean_std_dict, replace=True)

    def get_mean_std_dict_path(self):
        path = os.path.join(self.dataset_path, "mean_std_info.pkl")
        return path

    def get_config(self):
        from configs import config
        return config

    def get_selected_continuous_label_dim(self):
        dim = 0
        return dim

    @staticmethod
    def get_feature_dimension(config):
        feature_dimension = config['feature_dimension']
        return feature_dimension

    @staticmethod
    def get_multiplier(config):
        multiplier = config['multiplier']
        return multiplier

    @staticmethod
    def get_time_delay(config):
        time_delay = config['time_delay']
        return time_delay

    def init_randomness(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def init_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.high_performance_cluster:
            torch.cuda.set_device(self.gpu)

        torch.set_num_threads(self.cpu)

        return device

    def print_output(self, save_path):
        output_dir = os.sep.join(save_path.split(os.sep)[:-1])
        test_results = []
        val_results = []

        f = open(os.path.join(output_dir, 'summary.txt'), 'w')
        for fold in self.folds_to_run:
            folder = str(fold) + "_" + self.emotion
            csv_path = os.path.join(output_dir, folder, "training_logs.csv")
            with open(csv_path) as file:
                lines = file.readlines()
            best_epoch = int(float(lines[-2].split(",")[2]))
            for line in lines:
                try:
                    epoch = int(float(line.split(",")[1]))
                except:
                    epoch = -1
                if epoch == best_epoch:
                    val_ccc = round(float(line.split(",")[-1]), 3)
                    val_results.append(val_ccc)
            test_ccc = float(lines[-1].split(",")[-1])
            test_results.append(test_ccc)
            output = ""
            output += "fold_" + str(fold) + "________"
            output += "val" + "_" + str(round(val_ccc, 3)) + "________"
            output += "test" + "_" + str(round(test_ccc, 3))

            f.write(output + "\n")
            print(output)

        assert len(self.folds_to_run) == len(test_results)
        ccc_mean_val = np.mean(val_results)
        ccc_std_val = np.std(val_results)
        ccc_mean_test = np.mean(test_results)
        ccc_std_test = np.std(test_results)
        output = ""
        output += "Summary" + "________" + "val" + "_"
        output += str(round(ccc_mean_val, 3))
        output += "+-"
        output += str(round(ccc_std_val, 3))
        output += "________" + "test" + "_"
        output += str(round(ccc_mean_test, 3))
        output += "+-"
        output += str(round(ccc_std_test, 3))
        f.write(output)
        print(output)
        f.close()
