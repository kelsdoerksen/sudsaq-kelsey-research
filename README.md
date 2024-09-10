<div align="center">

# Deep Learning with UQ for Physical Model Bias of Surface Ozone

<p>
<b><a href="#-description">Description</a></b>
|
<b><a href="#-dataset">Dataset</a></b>
|
<b><a href="#-code-organization">Code Organization</a></b>
</p>

</div>

## 📄 Description
This work presents the data processing, model training, testing, and analysis for the purposes of surface ozone bias modelling with Deep Learning and Uncertainty Quantification. Air pollution is a global hazard, and as of 2023, 94% of the world’s population is exposed to unsafe pollution levels Sanchez-Triana (2023). This code implements an uncertainty-aware U-Net architecture to predict the Multi-mOdel Multi-cOnstituent Chemical data assimilation (MOMO-Chem) model’s surface ozone residuals (bias) using Bayesian and quantile regression1 methods for North America and Europe, with extensions to a Global analysis (WIP).

<p>

## 🌍 Dataset
The satellite data products used in this study are available from Google Earth Engine. The list of datasets used to generate the model feature space are below:

[MODIS Landcover](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1)

[Gridded Population of the World]([https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMCFG](https://developers.google.com/earth-engine/datasets/catalog/CIESIN_GPWv411_GPW_Population_Density))


## 📚 Code Organization
To run the pipeline, the following command is used:
```
python run_pipeline.py --epochs --optimizer --classes  --test-year --overfitting_test  --channels  --target  --region  --seed  --model_type  --data_dir --save_dir --val_percent --analysis_date --tag 
```
The available configurable parameters are:
* `--epochs`: Training Epochs
* `--optimizer`: U-Net optimizer
* `--classes`: Number of target classes
* `--test_year`: Year of test set data
* `--overfitting_test`: Specify if you would like to run quick experiment to ensure data can be overfit
* `--channels`: Number of channels in feature space
* `--target`: Name of target variable (currently supports bias)
* `--region`: Region of analysis. Currently supports NorthAmerica, Europe, Globe (WIP)
* `--seed`: Specify seed if deterministic experiment desired
* `--model_type`: Specify model type. Supports standard (no UQ), CQR and MC-Dropout
* `--data_dir`: Directory of stored data
* `--save_dir`: Results directory
* `--val_percent`: Desired percentage of training set to be set aside for validation
* `--analysis_date`: Test year
* `--tag`: Wandb experiment tag

The below folders host the following code:

`unet`: home of run_pipeline.py script and modules for loading dataset, training, testing.

`unet/data_processing`: all pre-processing scripts to generate feature space.

`unet/analysis`: scripts for post-processing results into figures and maps.
