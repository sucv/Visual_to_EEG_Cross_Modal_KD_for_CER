import cv2
import subprocess
import os
from utils.utils import copy_file


class VideoSplit(object):
    r"""
        A base class to  split video according to a list. For example, given
        [(0, 1000), (1200, 1500), (1800, 1900)] as the indices, the associated
        frames will be split and combined  to form a new video.
    """

    def __init__(self, input_filename, output_filename, trim_range):
        r"""
        The init function of the class.
        :param input_filename: (str), the absolute directory of the input video.
        :param output_filename:  (str), the absolute directory of the output video.
        :param trim_range: (list), the indices of useful frames.
        """

        self.input_filename = input_filename
        self.output_filename = output_filename

        self.video = cv2.VideoCapture(self.input_filename)

        # The frame count.
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        # The fps count.
        self.fps = self.video.get(cv2.CAP_PROP_FPS)

        # The size of the video.
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # The range to trim the video.
        self.trim_range = trim_range

        # The settings for video writer.
        self.codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(output_filename,
                                      self.codec, self.fps,
                                      (self.width, self.height), isColor=True)

    def jump_to_frame(self, frame_index):
        r"""
        Jump to a specific frame by its index.
        :param frame_index:  (int), the index of the frame to jump to.
        :return: none.
        """
        self.video.set(1, frame_index)

    def read(self, start, end, visualize):
        r"""
        Read then write the frames within (start, end) one frame at a time.
        :param start:  (int), the starting index of the range.
        :param end:  (int), the ending index of the range.
        :param visualize:  (boolean), whether to visualize the process.
        :return:  none.
        """

        # Jump to the starting frame.
        self.jump_to_frame(start)

        # Sequentially write the next end-start frames.
        for index in range(end - start):
            ret, frame = self.video.read()
            self.writer.write(frame)
            if ret and visualize:
                cv2.imshow('frame', frame)
                # Hit 'q' on the keyboard to quit!
                cv2.waitKey(1)

    def combine(self, visualize=False):
        r"""
        Combine the clips  into a single video.
        :param visualize: (boolean), whether to visualize the process.
        :return:  none.
        """

        # Iterate over the pair of start and end.
        for (start, end) in self.trim_range:
            self.read(start, end, visualize)

        self.video.release()
        self.writer.release()
        if visualize:
            cv2.destroyWindow('frame')


def change_video_fps(input_path, output_path, target_fps):
    r"""
    Alter the frame rate of a given video.
    :param videos:  (list),a list of videos to process.
    :param target_fps:  (float), the desired fps.
    :return: (list or str), the list  (or str if only one input video) of videos after the process.
    """
    output_video_list = []
    print("Changing video fps...")

    # If the new name already belongs to a file, then do nothing.
    if os.path.isfile(output_path):
        print("Skipped fps conversion for video {}!".format(output_path))
        pass

    # If not, call the ffmpeg tools to change the fps.
    # -qscale:v 0 can preserve the quality of the frame after the recoding.
    else:
        input_codec = " xvid "
        if ".mp4" in input_path:
            input_codec = " mp4v "
        command = "ffmpeg -i {} -filter:v fps=fps={} -c:v mpeg4 -vtag {} -qscale:v 0 {}".format(
            '"' + input_path + '"', str(target_fps), input_codec,
            '"' + output_path + '"')
        subprocess.call(command, shell=True)


def combine_annotated_clips(
        input_path,
        output_path,
        trim_range,
        direct_copy=False,
        visualize=False
):


    print("combining annotated clips...")

    # If the new name already belongs to a file, then do nothing.
    if os.path.isfile(output_path):
        print("Skipped video combination for video {}!".format(output_path))
        pass

    # If not, call the video combiner.
    else:
        if not direct_copy:
            video_split = VideoSplit(input_path, output_path, trim_range)
            video_split.combine(visualize)
        else:
            copy_file(input_path, output_path)


class OpenFaceController(object):
    def __init__(self, openface_config, output_directory):
        self.openface_config = openface_config
        self.output_directory = output_directory

    def get_openface_command(self):
        openface_path = self.openface_config['directory']
        input_flag = self.openface_config['input_flag']
        output_features = self.openface_config['output_features']
        output_action_unit = self.openface_config['output_action_unit']
        output_image_flag = self.openface_config['output_image_flag']
        output_image_size = self.openface_config['output_image_size']
        output_image_format = self.openface_config['output_image_format']
        output_filename_flag = self.openface_config['output_filename_flag']
        output_directory_flag = self.openface_config['output_directory_flag']
        output_directory = self.output_directory
        output_image_mask_flag = self.openface_config['output_image_mask_flag']

        command = openface_path + input_flag + " {input_filename} " + output_features \
                  + output_action_unit + output_image_flag + output_image_size \
                  + output_image_format + output_filename_flag + " {output_filename} " \
                  + output_directory_flag + output_directory + output_image_mask_flag
        return command

    def process_video(self, input_filename, output_filename):

        # Quote the file name if spaces occurred.
        if " " in input_filename:
            input_filename = '"' + input_filename + '"'

        command = self.get_openface_command()
        command = command.format(
            input_filename=input_filename, output_filename=output_filename)

        if not os.path.isfile(os.path.join(self.output_directory, output_filename + ".csv")):
            subprocess.call(command, shell=True)

