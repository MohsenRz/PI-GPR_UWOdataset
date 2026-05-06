## RAW_data

This directory is intended to store the raw dataset used in this project.

The original data are large files in SQLite format. Due to their size, they are excluded from version control via the `.gitignore` file in the root directory.

The scripts in this repository use preprocessed `.pkl` (pickle) files derived from the SQLite data. These files are also relatively large and are therefore not included in the repository.
You can use the `convert to pkl.py` file in this folder to get the pickle files needed. 

To run the code, you will need to:
- provide the required SQLite or `.pkl` files in this directory, or
- update the data path in the code to point to your local data location.
