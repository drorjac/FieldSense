# Project Structure

This project is organized into the following main folders:

# Adding Required Files

This dataset requires a large file to be manually added. Please follow the instructions below:

- **Required File:** `big_file.csv`
- **Location:** Place the file in this folder.
- **Download Link:** [Download big_file.csv](https://example.com/big_file)

Ensure the file is named correctly and placed here for the scripts to process it.

## 1. `open_datasets/`
- **Purpose:** Contains the raw datasets for each project. These datasets are provided as examples and include metadata and measurement files.
- **Contents:**
  - `meta_data` file: Describes the dataset's tools, features, and structure.
  - `raw_data` file: Sample measurements or data used as input for processing.
- **Note:** The full datasets can be accessed from the provided download links in each folder's `README.md`.

---


## 2. `our_datasets/`
- **Purpose:** Stores the processed datasets, transformed into the desired format for analysis or modeling.
- **Contents:**
  - Processed datasets that are ready for use.
  - Files are named consistently with the dataset they originate from.

---

## 3. `processing/`
- **Purpose:** Contains scripts and functions to read and process the datasets.
- **Contents:**
  - Dataset-specific scripts for handling raw data, such as cleaning, reshaping, and saving the data in a standardized format.
  - Shared tools for common data transformations, reusable across datasets.

---

## Workflow
1. **Add raw data**: Place raw data files in the respective subfolders.
2. **Process the data**: Use the scripts in `processing/` to clean and convert the data.
3. **Store processed data**: Save the processed datasets in `our_datasets/`.

For more information on each dataset, refer to the `meta_data` file in the corresponding folder.
