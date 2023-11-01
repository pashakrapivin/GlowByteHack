import pandas as pd
import holidays


# Функция извлечения признаков, для предсказаний на 1 день
def features_extractor_1d(dataset, target_encoder):
    features = pd.DataFrame()
    res_dataset = dataset.resample('1D')

    # Вчерашняя температура
    features['temp_pred'] = res_dataset.agg({'temp_pred': 'mean'})
    features['yst_temp_min'] = res_dataset.agg({'temp': 'min'}).shift(1)
    features['yst_temp_med'] = res_dataset.agg({'temp': 'median'}).shift(1)
    for i in range(3):
        features[f'yst_temp_mean{i+1}'] = res_dataset.agg({'temp': 'mean'}).shift(i+1)
    features['yst_temp_max'] = res_dataset.agg({'temp': 'max'}).shift(1)
    features['yst_temp_last'] = dataset.query('time==23')['temp'].shift(1)
    # Вчерашний таргет
    features['yst_target_min'] = res_dataset.agg({'target': 'min'}).shift(1)
    features['yst_target_med'] = res_dataset.agg({'target': 'median'}).shift(1)
    features['yst_target_mean'] = res_dataset.agg({'target': 'mean'}).shift(1)
    features['yst_target_max'] = res_dataset.agg({'target': 'max'}).shift(1)
    features['yst_target_last'] = dataset.query('time==23')['target'].shift(1)

    # Погода энкодед
    features[['weather_pred', 'yst_weather_fact_1']] = (
        target_encoder.transform(dataset[['weather_pred', 'weather_fact']]).resample('D').mean()
    )
    features['yst_weather_fact_1'] = features['yst_weather_fact_1'].shift(1)
    for i in range(1, 2):
        features[f'yst_weather_fact_{i+1}'] = features['yst_weather_fact_1'].shift(i+1)


    # Лаг
    for i in range(31):
        features[f'target_lag_{i+1}'] = res_dataset.agg({'target': 'sum'}).shift(i+1)


    # Дни
    features['week_day'] = features.index.dayofweek
    features['year'] = features.index.year
    features['month'] = features.index.month
    features['day'] = features.index.day
    #features['quarter'] = (features.index.month - 1) // 3 + 1

    # Выходные праздники
    #features['is_weekend'] = [1 if day == 6 or day == 0 else 0 for day in features.week_day]
    holidays_list = []
    for i in holidays.RUS(years=[2019, 2020, 2021, 2022, 2023]).items():
        holidays_list.append(str(i[0]))
    holidays_list = holidays_list + ['2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31']
    features['is_holiday'] = [1 if str(val).split()[0] in holidays_list else 0 for val in features.index]

    features = features.dropna()
    return features


# Функция извлечения признаков, для предсказаний на 1 час
def features_extractor(dataset, target_encoder, predict_1d):
    features = pd.DataFrame()

    features['time'] = dataset['time']

    # Температура
    features['temp_pred'] = dataset['temp_pred']
    features['temp_pred'] = features['temp_pred'].ffill()
    features['yst_temp'] = dataset['temp'].shift(24)
    features['yst_temp_mean'] = features.resample('1D').agg({'yst_temp': 'mean'})
    features['yst_temp_med'] = features.resample('1D').agg({'yst_temp': 'median'})

    # Погода
    features['weather_pred'] = dataset['weather_pred']
    features['weather_pred'] = features['weather_pred'].ffill()
    features['weather_fact'] = dataset['weather_fact'].shift(24)
    features[['weather_pred', 'weather_fact']] = (
        target_encoder.transform(features[['weather_pred', 'weather_fact']])
    )
    features = features.rename(columns={'weather_fact': 'yst_weather_fact'})
    features['yst_weather_fact_mean'] = features.resample('1D').agg({'yst_weather_fact': 'mean'})
    features['yst_weather_fact_med'] = features.resample('1D').agg({'yst_weather_fact': 'median'})

    # Лаг таргета на 7 дней
    for i in range(24, 192, 24):
        features[f'yst_target_{i}'] = dataset['target'].shift(i)

    features['yst_target_sum'] = features.resample('1D').agg({'yst_target_24': 'sum'})

    # Предикт на 1 день
    features = features.join(predict_1d)

    # Дни
    features['week_day'] = features.index.dayofweek
    features['year'] = features.index.year
    features['month'] = features.index.month
    features['day'] = features.index.day

    # Праздники
    holidays_list = []
    for i in holidays.RUS(years=[2019, 2020, 2021, 2022, 2023]).items():
        holidays_list.append(str(i[0]))
    holidays_list = holidays_list + ['2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31']
    features['is_holiday'] = [1 if str(val).split()[0] in holidays_list else 0 for val in features.index]

    features = features.dropna()
    return features


# Функция для извлечения таргета
def target_creator(dataset, features_idx):
    target = dataset['target']
    target = target.loc[features_idx.unique()]
    return target
