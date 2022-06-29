from scipy.signal import welch
from scipy.integrate import simps
import numpy as np
import mne


class GenericEegController(object):

    def __init__(self, filename, config):
        self.filename = filename
        self.buffer_sec = config['buffer_sec']
        self.frequency = config['sampling_frequency']
        self.window_sec = config['window_sec']
        self.step = int(config['hop_sec'] * self.frequency)
        self.interest_bands = config['interest_bands']
        self.channel_slice = config['channel_slice']
        self.eeg_feature_list = config['features']
        self.extracted_data = self.preprocessing()

    def calculate_bandpower(self, data):
        data_np = data[:].T
        power_spectram_densities = []
        start = 0
        end = int(start + self.frequency * self.window_sec)
        while end < data_np.shape[1]:
            bandpower = bandpower_multiple(data_np[:, start:end], sampling_frequence=self.frequency,
                                           band_sequence=self.interest_bands, window_sec=1, relative=True)
            power_spectram_densities.append(bandpower)
            start = start + self.step
            end = int(start + self.frequency * self.window_sec)
        power_spectram_densities = np.stack(power_spectram_densities)
        return power_spectram_densities

    def preprocessing(self):
        raw_data = self.read_data()
        channel_type_dictionary = self.get_channel_type_dictionary(raw_data)
        raw_data = self.set_channel_types_from_dictionary(raw_data, channel_type_dictionary)
        crop_range = self.get_crop_range_in_second(raw_data)
        cropped_raw_data = self.crop_data(raw_data, crop_range)
        cropped_eeg_raw_data = self.get_eeg_data(cropped_raw_data)
        filtered_data_np = self.filter_eeg_data(cropped_eeg_raw_data)
        average_referenced_data_np = self.average_reference(filtered_data_np)


        extracted_data = {}

        if "eeg_bandpower" in self.eeg_feature_list:
            bandpower = self.calculate_bandpower(average_referenced_data_np)
            extracted_data.update({'eeg_bandpower': bandpower})

        return extracted_data

    def produce_bandpower_array(self, band_power):
        # 32 channel
        num_bp_per_sample = len(self.interest_bands) * 32
        bp_array = np.reshape(band_power, (-1, num_bp_per_sample))
        return bp_array

    @staticmethod
    def set_interest_bands():
        interest_bands = [(0.3, 4), (4, 8), (8, 12), (12, 18), (18, 30), (30, 45)]
        return interest_bands

    @staticmethod
    def filter_eeg_data(data):
        r"""we'l
        Filter the eeg signal using lowpass and highpass filter.
        :return: (mne object), the filtered eeg signal.
        """
        filtered_eeg_data = data.copy().load_data().filter(l_freq=0.3, h_freq=45)
        return filtered_eeg_data

    @staticmethod
    def average_reference(data):
        average_referenced_data = data.copy().load_data().set_eeg_reference()
        return average_referenced_data[:][0].T

    def read_data(self):
        r"""
        Load the bdf data using mne API.
        :return: (mne object), the raw signal containing different channels.
        """

        raw_data = mne.io.read_raw_bdf(self.filename)

        return raw_data

    def get_channel_slice(self):
        r"""
        Assign a tag to each channel according to the dataset paradigm.
        :return:
        """
        channel_slice = {'eeg': slice(0, 32), 'ecg': slice(32, 35), 'misc': slice(35, -1)}
        return channel_slice

    def get_channel_type_dictionary(self, data):
        r"""
        Generate a dictionary where the key is the channel names, and the value
            is the modality name (such as eeg, ecg, eog, etc...)
        :return: (dict), the dictionary of channel names to modality name.
        """
        channel_type_dictionary = {}
        for modal, slicing in self.channel_slice.items():
            channel_type_dictionary.update({channel: modal
                                            for channel in data.ch_names[
                                                self.channel_slice[modal]]})

        return channel_type_dictionary

    @staticmethod
    def set_channel_types_from_dictionary(data, channel_type_dictionary):
        r"""
        Set the channel types of the raw data according to a dictionary. I did this
            in order to call the automatic EOG, ECG remover. But it currently failed. Need to check.
        :return:
        """
        data = data.set_channel_types(channel_type_dictionary)
        return data

    def get_crop_range_in_second(self, data):
        r"""
        Assign the stimulated time interval for cropping.
        :return: (list), the list containing the time interval.
        """
        crop_range = [[30. - self.window_sec / 2, data.times.max() - 30 + self.buffer_sec]]
        return crop_range

    @staticmethod
    def crop_data(data, crop_range):
        r"""
        Crop the signal so that only the stimulated parts are preserved.
        :return: (mne object), the cropped data.
        """
        cropped_data = []
        for index, (start, end) in enumerate(crop_range):

            if index == 0:
                cropped_data = data.copy().crop(tmin=start, tmax=end)
            else:
                cropped_data.append(data.copy().crop(tmin=start, tmax=end))

        return cropped_data

    @staticmethod
    def get_eeg_data(data):
        r"""
        Get only the eeg data from the raw data.
        :return: (mne object), the eeg signal.
        """
        eeg_data = data.copy().pick_types(eeg=True)
        return eeg_data


def bandpower_multiple(data, sampling_frequence, band_sequence, window_sec=None, relative=False):
    # Compute the modified periodogram (Welch)

    nperseg = window_sec * sampling_frequence

    freqs, psd = welch(data, sampling_frequence, nperseg=nperseg)

    freq_res = freqs[1] - freqs[0]

    band_powers = []

    for band in band_sequence:
        low, high = band
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integral approximation of the spectrum using Simpson's rule.
        band_power = simps(psd[:, idx_band], dx=freq_res)

        if relative:
            band_power /= simps(psd, dx=freq_res)

        band_powers.extend(band_power)

    band_powers = np.asarray(band_powers)
    return band_powers
