from utils.logger import ContinuousOutputHandler, ContinuousMetricsCalculator, PlotHandler
from utils.utils import ensure_dir

from torch import optim
import torch
from torch.nn.functional import l1_loss
import time
import copy
import os

import numpy as np

from tqdm import tqdm


class Trainer(object):
    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.model_name = kwargs['model_name']
        self.model = kwargs['models'].to(self.device)
        self.save_path = kwargs['save_path']
        self.fold = kwargs['fold']
        self.min_epoch = kwargs['min_epoch']
        self.max_epoch = kwargs['max_epoch']
        self.start_epoch = 0
        self.early_stopping = kwargs['early_stopping']
        self.early_stopping_counter = self.early_stopping
        self.scheduler = kwargs['scheduler']
        self.learning_rate = kwargs['learning_rate']
        self.min_learning_rate = kwargs['min_learning_rate']
        self.patience = kwargs['patience']
        self.criterion = kwargs['criterion']
        self.factor = kwargs['factor']
        self.w = kwargs['w']
        self.verbose = kwargs['verbose']
        self.milestone = kwargs['milestone']
        self.load_best_at_each_epoch = kwargs['load_best_at_each_epoch']

        self.best_epoch_info = None
        self.optimizer, self.scheduler = None, None
        if len(self.get_parameters()) != 0:
            self.init_optimizer_and_scheduler()

        self.batch_size = kwargs['batch_size']
        self.emotion = kwargs['emotion']
        self.emotion_dim = kwargs['emotion_dim']
        self.metrics = kwargs['metrics']
        self.save_plot = kwargs['save_plot']

        # For checkpoint
        self.fit_finished = False
        self.fold_finished = False
        self.resume = False
        self.time_fit_start = None

        self.train_losses = []
        self.validate_losses = []
        self.csv_filename = None

        self.feature_path = kwargs['feature_path']

    def init_optimizer_and_scheduler(self):
        self.optimizer = optim.Adam(self.get_parameters(), lr=self.learning_rate, weight_decay=0.0001)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=self.patience, factor=self.factor)

    def test(self, checkpoint_controller, **kwargs):
        kwargs['train_mode'] = False

        with torch.no_grad():
            self.model.eval()

            loss, result_dict = self.loop(**kwargs)
            print(result_dict['overall'])
            checkpoint_controller.save_log_to_csv(
                kwargs['epoch'], mean_train_record=None, mean_validate_record=None, test_record=result_dict['overall'])

        return loss, result_dict

    def loop(self, **kwargs):
        dataloader_dict, epoch, train_mode = kwargs['dataloader_dict'], kwargs['epoch'], kwargs['train_mode']
        partition = kwargs['partition']

        if train_mode:
            dataloader = dataloader_dict['train']
        elif epoch is None:
            dataloader = dataloader_dict[partition]
        else:
            dataloader = dataloader_dict['validate']

        running_loss = 0.0
        total_batch_counter = 0
        inputs = {}

        output_handler = ContinuousOutputHandler()
        continuous_label_handler = ContinuousOutputHandler()

        metric_handler = ContinuousMetricsCalculator(self.metrics, self.emotion,
                                                     output_handler, continuous_label_handler)

        for batch_idx, (X, trials, lengths, indices) in tqdm(enumerate(dataloader), total=len(dataloader)):

            knowledge_distillation = 0

            total_batch_counter += len(trials)

            for feature, value in X.items():
                inputs[feature] = X[feature].to(self.device)

            if "continuous_label" in inputs:
                labels = inputs.pop("continuous_label", None)
            if "visual_knowledge" in inputs:
                knowledge_distillation = 1
                teacher_knowledge = inputs.pop("visual_knowledge", None)

            if len(torch.flatten(labels)) == self.batch_size:
                labels = torch.zeros((self.batch_size, len(indices[0]), 1), dtype=torch.float32).to(self.device)

            if train_mode:
                self.optimizer.zero_grad()

            outputs, student_knowledge = self.model(inputs)

            loss = self.criterion(labels, outputs)

            if knowledge_distillation:
                distill_loss = self.w * l1_loss(teacher_knowledge, student_knowledge)
                loss += distill_loss

            running_loss += loss.mean().item()

            if train_mode:
                loss.backward()
                self.optimizer.step()

            output_handler.update_output_for_seen_trials(outputs.detach().cpu().numpy(), trials, indices, lengths)
            continuous_label_handler.update_output_for_seen_trials(labels.detach().cpu().numpy(), trials, indices, lengths)

        epoch_loss = running_loss / total_batch_counter

        output_handler.average_trial_wise_records()
        continuous_label_handler.average_trial_wise_records()

        output_handler.concat_records()
        continuous_label_handler.concat_records()

        # Compute the root mean square error, pearson correlation coefficient and significance, and the
        # concordance correlation coefficient.
        # They are calculated by  first concatenating all the output
        # and continuous labels to two long arrays, and then calculate the metrics.
        metric_handler.calculate_metrics()
        epoch_result_dict = metric_handler.metric_record_dict

        metric_handler.save_trial_wise_records(self.save_path, train_mode, epoch)

        if self.save_plot:
            # This object plot the figures and save them.
            plot_handler = PlotHandler(self.metrics, self.emotion, epoch_result_dict,
                                       output_handler.trialwise_records,
                                       continuous_label_handler.trialwise_records,
                                       epoch=epoch, train_mode=train_mode,
                                       directory_to_save_plot=self.save_path)
            plot_handler.save_output_vs_continuous_label_plot()

        return epoch_loss, epoch_result_dict

    def fit(self, dataloader_dict, checkpoint_controller, parameter_controller, knowledge_distillation=0):

        if self.verbose:
            print("------")
            print("Starting training, on device:", self.device)

        self.time_fit_start = time.time()
        start_epoch = self.start_epoch

        if self.best_epoch_info is None:
            self.best_epoch_info = {
                'model_weights': copy.deepcopy(self.model.state_dict()),
                'loss': 1e10,
                'ccc': -1e10
            }

        for epoch in np.arange(start_epoch, self.max_epoch):

            if self.fit_finished:
                if self.verbose:
                    print("\nEarly Stop!\n")
                break

            improvement = False

            if epoch in self.milestone or self.optimizer.param_groups[0]['lr'] < self.min_learning_rate:
                if hasattr(self.model, 'spatial'):
                    parameter_controller.release_param(self.model.spatial, epoch)
                    if parameter_controller.early_stop:
                        break

                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            time_epoch_start = time.time()

            if self.verbose:
                print("There are {} layers to update.".format(len(self.optimizer.param_groups[0]['params'])))

            # Get the losses and the record dictionaries for training and validation.
            train_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch, "partition": "train"}
            train_loss, train_record_dict = self.train(**train_kwargs)

            validate_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch, "partition": "validate"}
            validate_loss, validate_record_dict = self.validate(**validate_kwargs)

            if validate_loss < 0:
                raise ValueError('validate loss negative')

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)

            validate_ccc = validate_record_dict['overall']['ccc']

            if validate_ccc > self.best_epoch_info['ccc']:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_state_dict.pth"))

                improvement = True
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'ccc': validate_ccc,
                    'epoch': epoch,
                }

            if self.verbose:
                print(
                    "\n Fold {:2} Epoch {:2} in {:.0f}s || Train loss={:.3f} | Val loss={:.3f} | LR={:.1e} | Release_count={} | best={} | "
                    "improvement={}-{}".format(
                        self.fold,
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        validate_loss,
                        self.optimizer.param_groups[0]['lr'],
                        parameter_controller.release_count,
                        int(self.best_epoch_info['epoch']) + 1,
                        improvement,
                        self.early_stopping_counter))

                print(train_record_dict['overall'])
                print(validate_record_dict['overall'])
                print("------")

            checkpoint_controller.save_log_to_csv(
                epoch, train_record_dict['overall'], validate_record_dict['overall'])

            # Early stopping controller.
            if self.early_stopping and epoch > self.min_epoch:
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
                    self.fit_finished = True

            self.scheduler.step(metrics=validate_ccc)
            self.start_epoch = epoch + 1

            if self.load_best_at_each_epoch:
                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.fit_finished = True
        checkpoint_controller.save_checkpoint(self, parameter_controller, self.save_path)

        self.model.load_state_dict(self.best_epoch_info['model_weights'])

    def train(self, **kwargs):
        kwargs['train_mode'] = True
        self.model.train()

        loss, result_dict = self.loop(**kwargs)
        return loss, result_dict

    def validate(self, **kwargs):

        kwargs['train_mode'] = False
        with torch.no_grad():
            self.model.eval()
            loss, result_dict = self.loop(**kwargs)
        return loss, result_dict

    def extract(self, **kwargs):
        kwargs['train_mode'] = False

        with torch.no_grad():
            self.model.eval()

        self.extract_loop(**kwargs)

    def extract_loop(self, **kwargs):
        partition = kwargs['partition']
        dataloader = kwargs['dataloader_dict'][partition]
        inputs = {}

        for batch_idx, (X, trials, lengths, indices) in tqdm(enumerate(dataloader), total=len(dataloader)):

            for feature, value in X.items():
                if "label" in feature:
                    continue
                inputs[feature] = X[feature]

            inputs_segment = {}
            feature = []
            for modal, value in inputs.items():
                trial_length = value.shape[1]

                step = 500
                num_window = int(np.ceil(trial_length / step))
                for i in range(num_window):
                    start = i * step
                    end = (i + 1) * step

                    data_segment = value[:, start:end, :, :, :]
                    inputs_segment[modal] = data_segment.to(self.device)
                    _, feature_segment = self.model(inputs_segment)

                    feature_segment = feature_segment.squeeze(0)
                    feature_segment = feature_segment.detach().cpu().numpy()
                    feature.append(feature_segment)

            feature = np.vstack(feature)
            feature_path = os.path.join(self.feature_path, trials[0], "visual_knowledge.npy")
            ensure_dir(feature_path)
            np.save(feature_path, feature)

    def get_parameters(self):
        r"""
        Get the parameters to update.
        :return:
        """
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        return params_to_update