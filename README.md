# CMSHGCaloChallenge


The challenge uses a 70/30 train/test split.
To ensure consistency we specify which files are to be used for which.
Users are welcome to split the training set into a train/validation split as they
wish.
The files from the testing set must not be used for training, validation or any sort of optimization. 

Lists detailing which files belong in each category are given in `datasets/`.

Photon datasets
Train/val (`photon_files_train.txt`) : 0-244
Test (`photon_files_test.txt`) : 245-350


Pion datasets
Train/val (`pion_files_train.txt`) : 0-249
Test (`pion_files_test.txt`) : 250-360


evaluation metrics are computed with the `hgcal_metrics.py` script.
It can be used like:
`python hgcal_metrics.py -c CONFIG.json -g GENERATED_FILES.txt -p PLOT_DIR/ -d DATA_DIR/`

Where `CONFIG.json` is a configuration file with details of the dataset. Either
`config_HGCal_pions.json` or `config_HGCAL_photons.json` 

`GENERATED_FILES.txt` is a text file with the paths to showers to be evaluated (from your model).

`PLOT_DIR/` is a directory to put the evaluation plots as well as a text file
with the numeric metrics.

`DATA_DIR/` is a directory where the Geant showers are stored.


An example usage would be
`python hgcal_metrics.py -c config_HGCal_pions.json -g datasets/HGCal_central_2024_pions_eval_test.txt -p plots/eval_test/ -d  /uscms_data/d3/oamram/HGCal/HGCal_central_2024_pions/`


Note that computing all of the features for evaluation takes quite some time
for the pion datasets. To avoid this significant overhead, the script saves
these computed features once for each input file with the same name extended
with .feat.npz. Subsequent runs will then use these pre-computed features if such
a file exists. To force a recreation of these files you can use the
`--reprocess` flag

The evaluation can optionally be done without the inclusion of the sparsity
feature with the `--no_sparse` flag.

If one only wants to run some partial metrics the `--mode` flag can be used
with options `hist, cls, fpd, all` (all is the default).
