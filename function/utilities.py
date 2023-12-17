import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from numba import njit
from scipy.ndimage import gaussian_filter1d, median_filter, maximum_filter


def get_datasets(top_directory, keywords=[], exclude=[]):
    folders = identify_folders(top_directory, ['dpf'])
    datasets = []
    for folder in folders:
        datasets += identify_folders(folder, keywords=keywords, exclude=exclude)
    return datasets
    
def identify_files(path, keywords=None, exclude=None):
    items = os.listdir(path)
    if keywords is None:
        keywords = []
    if exclude is None:
        exclude = []
    files = []
    for item in items:
        if all(keyword in item for keyword in keywords):
            if any(excluded in item for excluded in exclude):
                pass
            else:
                files.append(item)
    files.sort()
    return files

def identify_folders(path, keywords=None, exclude=None):
    initial_folders = [f.path for f in os.scandir(path) if f.is_dir()]
    if keywords is None:
        keywords = []
    if exclude is None:
        exclude = []
    folders = []
    for folder in initial_folders:
        if all(keyword in folder for keyword in keywords):
            if any(excluded in folder for excluded in exclude):
                pass
            else:
                folders.append(folder)
    for i in range(len(folders)):
        folders[i] += '/'
    folders.sort()
    return folders
    

class TailAnalysis:

    def __init__(self, tail_angles, fps, sigma_median=3, sigma_gaussian=1):
        
        self.fps = fps
        
        self.angles = self.filter_angles(tail_angles, sigma_median=sigma_median, sigma_gaussian=sigma_gaussian)
        for i in range(tail_angles.shape[0]):
            self.angles[i, :] -= np.mean(self.angles[i, :])
            
        self.curvature = np.sum(self.angles, axis=0)
        self.curvature = self.curvature - np.mean(self.curvature)

        self.velocity = np.abs(np.diff(self.curvature))
        self.t = np.linspace(0, self.angles.shape[1]/self.fps, self.angles.shape[1], endpoint=False)
        self.baseline = np.mean(np.sum(self.angles, axis=0))
        self.clusters = []

    def detect_swim_bouts(self, threshold=0.5, window_min=600, window_max=30, min_duration=45):

        # Converting to seconds
        w_min = int(self.fps * window_min / 1000)
        w_max = int(self.fps * window_max / 1000)
        min_duration = int(self.fps * min_duration / 1000)
        
        maxcurve = maximum_filter(subtract_local_minima(np.abs(self.curvature), w_min), w_max)
        plateaus = find_plateaus(maxcurve > threshold)
        plateaus = plateaus[np.sum(plateaus, axis=1) >= min_duration, :]

        events = np.sum(plateaus, axis=0)
        onsets = find_onsets(events)
        offsets = find_offsets(events)

        self.onsets, self.offsets, self.events = onsets, offsets, events
        self.compile_swim_bouts()

    def remove_low_amplitude_events(self, threshold=1):
        onsets, offsets = [], []
        for i, onset in enumerate(self.onsets):
            if np.sum(np.abs(self.curvature[onset:self.offsets[i]]) > threshold) == 0:
                self.events[onset:self.offsets[i]] = 0
            else:
                onsets.append(onset)
                offsets.append(self.offsets[i])
        self.onsets = onsets
        self.offsets = offsets
        self.compile_swim_bouts()

    def compile_swim_bouts(self):
        bouts = []
        for i in range(len(self.onsets)):
            bouts.append(self.angles[:, self.onsets[i]:self.offsets[i]])
        self.swim_bouts = bouts

    @staticmethod
    def filter_angles(angles, sigma_median=3, sigma_gaussian=1):
        filtered = np.copy(angles)
        for i in range(filtered.shape[0]):
            filtered[i, :] = gaussian_filter1d(
                median_filter(median_filter(filtered[i, :], sigma_median),
                              sigma_median),
                sigma_gaussian)
        return filtered

@njit
def subtract_local_minima(signal, window):
    subtracted = np.copy(signal)
    for i in range(len(signal)):
        if i > window:
            subtracted[i] -= np.min(signal[i-window:i])
    return subtracted

def find_plateaus(binary_vector):
    onsets = find_onsets(binary_vector.astype('float'))
    plateaus = np.zeros((len(onsets), len(binary_vector)))
    for i, onset in enumerate(onsets):
        j = onset
        plateaus[i, j] = 1
        while plateaus[i, j] == 1:
            j += 1
            plateaus[i, j] = binary_vector[j]
    return plateaus

def find_offsets(binary_vector):
    return np.where(np.append([0], np.diff(binary_vector.astype('float'))) < 0)[0]

def find_onsets(binary_vector):
    return np.where(np.append([0], np.diff(binary_vector.astype('float'))) > 0)[0]
