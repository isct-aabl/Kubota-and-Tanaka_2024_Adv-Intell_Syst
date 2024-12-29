# Kubota-and-Tanaka_2024_Adv-Intell_Syst
Data and Scripts of the paper entitled "Machine learning-based wind classification by wing deformation in biomimetic flapping robots: Biomimetic flexible structures improve wind sensing"

# Machine Learning-Based Data Classification Algorithm

This repository contains algorithms for supervised learning and classification of data using machine learning, along with related scripts and configuration files.

---

## Folder Structure

The dataset used in this algorithm follows the folder structure below (abstract representation):

```plaintext
data/
├── Wing_A/               # Data for Wing_A experiments
│   ├── training_data/
│   │   ├── 2_split_data/
│   │   │   └── raw_data/   # Raw training datasets
│   │   └── 3_training_data/
│   │       └── raw_data/   # Datasets ready for training
│   └── 1_original_data/
│       └── flapping/       # Contains subdirectories for each wind direction
│           ├── 0deg/       # Deformation data for 0 degrees wind direction
│           ├── 15deg/
│           ├── 30deg/
│           ├── 45deg/
│           ├── 60deg/
│           ├── 75deg/
│           └── 90deg/
├── Wing_A-2/             # Data for Wing_A-2 experiments (similar structure to Wing_A)
└── Wing_B/               # Data for Wing_B experiments (similar structure to Wing_A)
```

## How to Run the Algorithm

1. Request the dataset by contacting Prof. Tanaka (tanaka.h.cb@m.titech.ac.jp).
2. After receiving the dataset, place the `data` folder at the same level as the `scripts`, `models`, and `settings` folders in this repository:

   Project structure:
   ├── models/
   ├── scripts/
   ├── settings/
   ├── data/       # Add the received `data` folder here
   ├── .gitignore
   └── README.md

3. Open the `settings/general_variables.py` file and update the value of `setting_yaml_file_base` according to the specific wing configuration you want to measure. For example:
   
   ```python
   # Example in general_variables.py
   setting_yaml_file_base = "Wing_A"  # Choose from "Wing_A", "Wing_A-2", or "Wing_B"
   ```

4. Execute the relevant .ipynb files to run the experiments.

- 1a_split_raw_data.ipynb
- 1b_split_flatten_data.ipynb
- 2a_analyze_micro_data.ipynb
- 2b_analyze_macro_whole_data.ipynb
- 2c_analyze_macro_split_data.ipynb
- 3_train_CNN_model_efficiently.ipynb
- 3_train_CNN_model.ipynb
- 4_manage_CNN_model.ipynb
- 4b_manage_CNN_model_estimate_efficiently.ipynb

# Contact

For questions or data requests, please contact:

Prof. Tanaka

tanaka.h.cb@m.titech.ac.jp

