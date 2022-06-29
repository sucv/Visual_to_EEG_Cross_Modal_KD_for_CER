config = {

    "root_directory": r"F:\mahnob", # Specify this
    "output_root_directory": r"F:\mahnob_pr_processed", # Specify this

    "extract_continuous_label": 1,

    "trim_video": 1,
    "load_trimmed_video": 0,
    "trimmed_video_folder": "trimmed_video",

    "change_video_fps": 1,
    "load_fps_changed_video": 0,
    "fps_changed_video_folder": "fps_changed_video",
    "target_fps": 64,

    "crop_align_face": 1,
    "load_cropped_aligned_facial_image": 0,
    "cropped_aligned_folder": "cropped_aligned",
    "video_size": 48,

    "extract_facial_landmark": 1,
    "facial_landmark_folder": "cropped_aligned",

    "extract_eeg": 1,
    "eeg_folder": "eeg",
    "eeg_config": {
        "sampling_frequency": 256,
        "window_sec": 2,
        "hop_sec": 0.25,
        "buffer_sec": 5,
        "num_electrodes": 32,
        "interest_bands": [(0.3, 4), (4, 8), (8, 12), (12, 18), (18, 30), (30, 45)],
        "channel_slice": {'eeg': slice(0, 32), 'ecg': slice(32, 35), 'misc': slice(35, -1)},
        "features": ["eeg_bandpower"],
    },

    "save_npy": 1,
    "npy_folder": "compacted_48",


    "emotion_list": ["Valence"],
    "raw_data_folder": "Sessions",

    "multiplier": {
        "video": 16,
        "visual_knowledge": 16,
        "eeg_bandpower": 1,
        "landmark": 16,
        "continuous_label": 1,
    },

    "feature_dimension": {
        "video": (48, 48, 3),
        "visual_knowledge": (128,),
        "eeg_bandpower": (192,),
        "continuous_label": (1,),
        "landmark": (139,),
    },

    "openface": {
        "directory": "D:\\OpenFace_2.2.0_win_x64\\FeatureExtraction",  # Specify this
        "input_flag": " -f ",
        "output_features": " -2Dfp ",
        "output_action_unit": " -aus ",
        "output_image_flag": " -simalign ",
        "output_image_format": " -format_aligned jpg ",
        "output_image_size": " -simsize 48 ",
        "output_image_mask_flag": " -nomask ",
        "output_filename_flag": " -of ",
        "output_directory_flag": " -out_dir "
    },
}
