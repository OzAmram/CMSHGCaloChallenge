# CMSHGCaloChallenge


The challenge uses a 70/30 train/test split.
To ensure consistency, we specify the train and test file lists in [`datasets/`](./datasets).
Users are welcome to split the training set into a train/validation split as they wish.
The files from the testing set must not be used for training, validation, or any sort of optimization.

Photon datasets:
* Train/val (`photon_files_train.txt`) : 0-244
* Test (`photon_files_test.txt`) : 245-350

Pion datasets:
* Train/val (`pion_files_train.txt`) : 0-249
* Test (`pion_files_test.txt`) : 250-360

Evaluation metrics are computed with the `hgcal_metrics.py` script; to run it:
```
python hgcal_metrics.py -c CONFIG.json -g GENERATED_FILES.txt -p PLOT_DIR/ -d DATA_DIR/
```
where:
* `CONFIG.json` is a configuration file with details of the dataset (either `config_HGCal_pions.json` or `config_HGCAL_photons.json`)
* `GENERATED_FILES.txt` is a text file with the paths to showers to be evaluated (from your model).
* `PLOT_DIR/` is a directory to put the evaluation plots as well as a text file with the numeric metrics.
* `DATA_DIR/` is a directory where the Geant showers are stored.

An example usage would be:
```
python hgcal_metrics.py -c config_HGCal_pions.json -g datasets/HGCal_central_2024_pions_eval_test.txt -p plots/eval_test/ -d  /uscms_data/d3/oamram/HGCal/HGCal_central_2024_pions/
```
