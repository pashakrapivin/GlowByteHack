import sys
import os
import pandas as pd
import joblib
import warnings
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
from features_extractor import features_extractor, target_creator

warnings.simplefilter('ignore', category=FutureWarning)


# Проверка аргумента и открытие файла
sys.exit(1) if len(sys.argv) != 2 else None

file_path = sys.argv[1]

if not os.path.exists(file_path):
    print(f"Файл '{file_path}' не существует.")
    sys.exit(1)


df = pd.read_csv(file_path, index_col='date', parse_dates=True)


# Получаем из датасета features и target
target_encoder = joblib.load('../files/target_encoder_1d')
features = features_extractor(df, target_encoder)
target = target_creator(df, features.index)


# Загружаем модель, получаем предикт
lgbm_pipeline = joblib.load('../files/lgbm_pipeline_1d')
predict = lgbm_pipeline.predict(features)


# Импортируем предикт в файл
predictions = pd.DataFrame()
predictions['date'] = features.index
predictions['predict'] = predict
predictions.to_csv('predictions.csv', index=False)

print(f'MAE: {mean_absolute_error(target, predict)}\n'
      f'MAPE: {mean_absolute_percentage_error(target, predict)}\n'
      f'r2: {r2_score(target, predict)}')

