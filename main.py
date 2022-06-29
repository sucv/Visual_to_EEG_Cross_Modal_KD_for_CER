import sys
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Say hello')

    # ####### Only uncomment one group below ############
    # # Group 1: Train the teacher and extract the visual knowledge
    # parser.add_argument('-modality', default=["video", "continuous_label"], nargs="*", help='video, eeg_bandpower, landmark')
    # parser.add_argument('-extract_feature', default=1, type=int, help='Extract dark knowledge from the teacher?')
    # parser.add_argument('-use_kd_on_student', default=0, type=int, help='Extract dark knowledge from the teacher?')
    # parser.add_argument('-w', default=0, type=float, help='The weight of the L1 loss for the knowledge distillation.')

    # # Group 2: Train the standalone student without the supervision of the teacher
    # parser.add_argument('-modality', default=["eeg_bandpower", "continuous_label"], nargs="*", help='video, eeg_bandpower, landmark')
    # parser.add_argument('-extract_feature', default=0, type=int, help='Extract dark knowledge from the teacher?')
    # parser.add_argument('-use_kd_on_student', default=0, type=int, help='Extract dark knowledge from the teacher?')
    # parser.add_argument('-w', default=0, type=float, help='The weight of the L1 loss for the knowledge distillation.')
    #
    # Group 3: Train the student with the teacher's visual knowledge
    parser.add_argument('-modality', default=["eeg_bandpower", "continuous_label"], nargs="*", help='video, eeg_bandpower, landmark')
    parser.add_argument('-extract_feature', default=0, type=int, help='Extract dark knowledge from the teacher?')
    parser.add_argument('-use_kd_on_student', default=1, type=int, help='Extract dark knowledge from the teacher?')
    parser.add_argument('-w', default=1, type=float, help='The weight of the L1 loss for the knowledge distillation.')
    ####### Only uncomment one group above ############

    # Specify the experiment instance
    parser.add_argument('-experiment_name', default="VtE_KD", help='The experiment name.')
    parser.add_argument('-gpu', default=0, type=int, help='Which gpu to use?') # Specify this
    parser.add_argument('-cpu', default=1, type=int, help='How many threads are allowed?') # Specify this
    parser.add_argument('-high_performance_cluster', default=0, type=int, help='On high-performance server or not? Such as Colab where you cannot specify GPU freely.') # Specify this
    parser.add_argument('-stamp', default='define_the_instance', type=str, help='To indicate different experiment instances') # Specify this
    parser.add_argument('-resume', default=0, type=int, help='Resume from checkpoint?') # Specify this
    parser.add_argument('-seed', default=0, type=int)

    parser.add_argument('-case', default='loso', type=str, help='trs: trial-wise shuffling, loso.: leave-one-subject-out.') # Specify this

    parser.add_argument('-debug', default=0, type=str, help='When debug=1, only fold x 1 trials would be loaded. And the epoch is set to 1 for a quick run-over.') # Specify this

    # Calculate mean and std for each modality?
    parser.add_argument('-calc_mean_std', default=1, type=int, help='Calculate the mean and std and save to a pickle file')
    parser.add_argument('-cross_validation', default=1, type=int)
    parser.add_argument('-folds_to_run', default=["all"], nargs="+", type=int, help='Which fold(s) to run in this session?')

    parser.add_argument('-dataset_path', default='/home/zhangsu/dataset/mahnob3', type=str,
                        help='The root directory of the dataset.') # Specify this
    parser.add_argument('-dataset_folder', default='compacted_48', type=str,
                        help='The root directory of the dataset.') # Specify this
    parser.add_argument('-load_path', default='/home/zhangsu/Visual_to_EEG_KD_for_CER/load', type=str, help='The path to load the backbones.') # Specify this
    parser.add_argument('-save_path', default='/home/zhangsu/Visual_to_EEG_KD_for_CER/save', type=str, help='The path to save the trained models ') # Specify this
    parser.add_argument('-python_package_path', default='/home/zhangsu/Visual_to_EEG_KD_for_CER', type=str, help='The path to the entire repository.') # Specify this
    parser.add_argument('-save_model', default=1, type=int, help='Whether to save the models?')

    # Models
    parser.add_argument('-model_name', default="tcn", help='Model: tcn, lstm') # Specify this
    parser.add_argument('-backbone_mode', default="ir", help='Mode for resnet50 backbone: ir, ir_se')
    parser.add_argument('-backbone_state_dict', default="res50_ir_0.874", help='The filename for the backbone state dict.')
    parser.add_argument('-cnn1d_channels', default=[128, 128], nargs="+", type=int, help='The specific epochs to do something.')
    parser.add_argument('-cnn1d_kernel_size', default=3, type=int, help='The size of the 1D kernel for temporal convolutional networks.')
    parser.add_argument('-cnn1d_dropout', default=0.1, type=float, help='The dropout rate.')

    parser.add_argument('-lstm_embedding_dim', default=64, type=int, help='Dimensions for LSTM feature vectors.')
    parser.add_argument('-lstm_hidden_dim', default=64, type=int, help='The size of the 1D kernel for temporal convolutional networks.')
    parser.add_argument('-lstm_dropout', default=0.4, type=float, help='The dropout rate.')

    parser.add_argument('-learning_rate', default=1e-5, type=float, help='The initial learning rate.')
    parser.add_argument('-min_learning_rate', default=1e-6, type=float, help='The minimum learning rate.')
    parser.add_argument('-min_num_epochs', default=0, type=int, help='The minimum epoch to run at least.')
    parser.add_argument('-time_delay', default=0, type=float, help='The time delay between input and label, in seconds.')
    parser.add_argument('-early_stopping', default=20, type=int, help='If no improvement, the number of epoch to run before halting the training')

    # Groundtruth settings
    parser.add_argument('-emotion', default="valence", type=str, help='The emotion dimension to analysis.')
    parser.add_argument('-metrics', default=["rmse", "pcc", "ccc"], nargs="*", help='The evaluation metrics.')

    # Dataloader settings
    parser.add_argument('-window_length', default=96, type=int, help='The length in second to windowing the data.')
    parser.add_argument('-hop_length', default=32, type=int, help='The step size or stride to move the window.')

    parser.add_argument('-frame_size', default=48, type=int, help='The size of the images.')
    parser.add_argument('-crop_size', default=40, type=int, help='The size to conduct the cropping.')
    parser.add_argument('-batch_size', default=2, type=int)

    # Scheduler and Parameter Control
    parser.add_argument('-scheduler', default='plateau', type=str, help='plateau, cosine')
    parser.add_argument('-patience', default=5, type=int, help='Patience for learning rate changes.')
    parser.add_argument('-factor', default=0.5, type=float, help='The multiplier to decrease the learning rate.')
    parser.add_argument('-load_best_at_each_epoch', default=1, type=int, help='Whether to load the best models state at the end of each epoch?')

    parser.add_argument('-save_plot', default=0, type=int,
                        help='Whether to plot the session-wise output/target or not?')

    args = parser.parse_args()
    sys.path.insert(0, args.python_package_path)

    from experiment import Experiment

    exp = Experiment(args)
    exp.prepare()
    exp.run()
