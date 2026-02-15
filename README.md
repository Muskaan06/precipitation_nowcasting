# precipitation_nowcasting
Creating benchmark nowcasting dataset for Indian regions

## Repo structure explanation
**data_images** : contains example precipitation mapping of different regions
**src**: contains the main code for dataset generation. The file _crop_and_set_ reads the raw data folder and generated a set of 24 files for each region. The file _random_selection_ selects a set only if contains files having acceptable amount of rainfall.
**utils**: contains codes to view data and some extra files
