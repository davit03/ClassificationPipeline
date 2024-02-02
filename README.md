# ClassificationPipeline
> ML ACA Project

The project includes four file:
-   `requirements.txt`
-   `preprocessor.py`
-   `model.py` 
-   `run_pipeline.py` 

Additionally, there are 1 saved model
-   `model.pkl` - trained ML model


## Train mode
In order to train a model on your data, use the following command 

```python
python3 run_pipeline.py --data_path Survival_dataset.csv
```

If you want to save the trained models (both preprocessor and the ML model), use the following command:
```python
python3 run_pipeline.py --save_models --data_path Survival_dataset.csv
```

## Test mode
In order to run the script in the test mode, use the following command:
```python
python3 run_pipeline.py --test --data_path Survival_dataset.csv
```
