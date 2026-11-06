# Melanoma Classification
Exploring transfer learning and staged fine-tuning of different architectures for melanoma classification

## Project Organization

```
├── LICENSE            
├── Makefile           
├── README.md          
├── data
│   ├── external       
│   ├── interim        
│   ├── isic2020-224   <- ISIC 2020 competition training set
│   ├── ph2_cache      <- external PH2 dermoscopy test set
│   ├── processed
│   └── raw            
│
├── docs               
│
├── models             <- trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- jupyter notebooks. 
│
├── pyproject.toml     <- project configuration file with package metadata for melanoma_cls
│
├── references         
│
├── reports            
│   └── figures        
│
├── requirements.txt   <- the requirements file generated with `pip freeze > requirements.txt`
│
├── setup.cfg          
│
└── melanoma_cls   
    │
    ├── __init__.py             
    │
    ├── config.py               <- useful variables and configuration
    │
    ├── dataset.py              <- scripts to download or generate data
    │
    ├── features.py             <- code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- code to run model inference with trained models          
    │   └── train.py            <- code to train models
    │
    └── plots.py                <- code to create visualizations
```

--------

