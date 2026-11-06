# Melanoma Classification
Exploring transfer learning and staged fine-tuning of different architectures for melanoma classification

## Project Organization

```
├── LICENSE            
├── Makefile           
├── README.md          
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. 
│
├── pyproject.toml     <- Project configuration file with package metadata for melanoma_cls
│
├── references         
│
├── reports            
│   └── figures        
│
├── requirements.txt   <- The requirements file generated with `pip freeze > requirements.txt`
│
├── setup.cfg          
│
└── melanoma_cls   
    │
    ├── __init__.py             
    │
    ├── config.py               <- Useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

