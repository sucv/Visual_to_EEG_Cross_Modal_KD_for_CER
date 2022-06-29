## Data Acquisition

The complete raw data of MAHNOB-HCI database is available at [this link](https://mahnob-db.eu/hci-tagging/).

The continuous labels are available at [this link](https://github.com/soheilrayatdoost/ContinuousEmotionDetection/tree/master/data).

Our trained visual backbone is available at [this link](https://drive.google.com/file/d/1izzZNtRIGchbyhf-aiTKB950IH1ZeJ4C/view?usp=sharing).

## Conda Environment Creation

First, please kindly install PyTorch. Then install the following packages by:

```
pip install pandas opencv-python tqdm mne pillow
```

## OpenFace Installation

Please download and install OpenFace from [this link](https://github.com/TadasBaltrusaitis/OpenFace/releases/tag/OpenFace_2.2.0). The installation instruction can be found at [this link](https://github.com/TadasBaltrusaitis/OpenFace/wiki).


## Preprocessing

In `configs.py`, please specify the follows.
- `root_directory`, which is the root directory of the MAHNOB-HCI data. The directory should includes two folders, named `Sessions` and `Subjects`.
- `output_root_directory`, where to store the processed data.
- `openface directory`, the directory containing the executable `FeatureExtraction`. In Windows it is an `.exe` file in the installed OpenFace root directory.

Then, run `preprocessing.py`. An IDE (e.g., PyCharm) instead of command line is required to do this, or the package import may raise errors. Sorry for the rush programming.

Note that the preprocessing may be convenient to run in a local personal PC with Windows system, rather than on a remote Linux server. It may take 1 day to complete.

## Experiment

In `main.py`, please specify the follows.

- `stamp`: name the experiment. It determines the folder to save the trained teacher, knowledge, student, training logs, and checkpoint.
- `case`: choose to run `loso` for leave-one-subject or `trs` for trial-wise random shuffling.
- `debug`: set to 1 to quickly run everything else 0.

- `dataset_path`: the path to the processed data.
- `dataset_folder`: the folder storing the npy files. It's a sub-folder of `dataset_path`.
- `load_path`: where to load the visual backbone.
- `save_path`: where to save the experiment output, including trained teacher, knowledge, student, training logs, and checkpoint.
- `python_package_path`: the path to this code.

- `model_name`: choose between `tcn` and `lstm` for temporal modeling.

Once set, uncomment one of the three groups, which was well-commented in `main.py`. 
- Group 1 would train the visual teacher and then extract/save the visual knowledge.
- Group 2 would train the standalone student without the supervision of the teacher.
- Group 3 would train the student with the teacher's visual knowledge.

Finally, run `main.py`. It is okay using either command line or IDE. About 4~5 G VRAM is required.

## Citation

Please consider citing our paper if you found the humble code helpful. 

```
@article{zhang2022visual,
  title={Visual-to-EEG cross-modal knowledge distillation for continuous emotion recognition},
  author={Zhang, Su and Tang, Chuangao and Guan, Cuntai},
  journal={Pattern Recognition},
  pages={108833},
  year={2022},
  publisher={Elsevier}
}
```