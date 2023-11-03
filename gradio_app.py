import pandas as pd
import gradio as gr
from plotly.callbacks import Points, BoxSelector
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

from stage_2 import features_extractor

warnings.simplefilter('ignore', category=FutureWarning)


def get_predictions(file, stage):
    df = pd.read_csv(file.name, index_col='date', parse_dates=True)

    # Получаем из датасета features и target для предсказания дневной модели
    target_encoder_1d = joblib.load('files/target_encoder_1d')
    features_1d = features_extractor.features_extractor_1d(df, target_encoder_1d)

    if stage == 'Этап 1. Предсказание на сутки.':
        # Загружаем модель, получаем предикт на 1 сутки
        lgbm_pipeline_1d = joblib.load('files/lgbm_pipeline_1d')
        predict = lgbm_pipeline_1d.predict(features_1d)

        # Импортируем предикт в файл
        predictions = pd.DataFrame()
        predictions['date'] = features_1d.index.astype(str)
        predictions['predict'] = predict
        predictions.to_csv('predictions.csv', index=False)

        # Берем таргет для отрисовки на графике и расчета метрики
        target = features_extractor.target_creator(df.resample('1D').sum(), features_1d.index)
        ids = predictions['date']

    elif stage == 'Этап 2. Предсказание на час.':
        # Загружаем модель, получаем предикт на 1 сутки
        stack_pipeline_1d = joblib.load('files/stack_pipeline_1d')
        predict_1d = pd.DataFrame(
            stack_pipeline_1d.predict(features_1d),
            index=features_1d.index,
            columns=['target_sum_pred']
        )

        # Получаем из датасета features и target для предсказания почасовой модели
        target_encoder = joblib.load('files/target_encoder_1h')
        features = features_extractor.features_extractor(df, target_encoder, predict_1d)

        # Загружаем модель, получаем предикт на 1 час
        lgbm_pipeline_1h = joblib.load('files/lgbm_pipeline_1h')
        predict = lgbm_pipeline_1h.predict(features)

        # Импортируем предикт в файл
        predictions = pd.DataFrame()
        predictions['datetime'] = features.index.astype(str) + ' ' + features['time'].astype(str) + ':00:00'
        predictions['predict'] = predict
        predictions.to_csv('predictions.csv', index=False)

        target = features_extractor.target_creator(df, features.index)

        ids = predictions['datetime']

    # График
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ids,
        y=target,
        opacity=0.7,
        line=dict(color='#1E90FF'),
        name='True'
    ))
    fig.add_trace(go.Scatter(
        x=ids,
        y=predictions['predict'],
        opacity=0.7,
        line=dict(color='orange'),
        name='Predict'
    ))
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=0.9
    ))
    fig.show()

    # Расчет метрик
    metrics = (f'MAE: {mean_absolute_error(target, predict):.3f} '
               f'r2: {r2_score(target, predict):.3f} '
               f'MAPE: {mean_absolute_percentage_error(target, predict):.3f} ')

    return predictions, 'predictions.csv', fig, metrics


# Создание интерфейса Gradio
with gr.Blocks(title='GlowByte Hackaton') as iface:
    with gr.Row():
        gr.Markdown(
            """
            # <center> GlowByte Hackaton <center>
            <center> <span style="font-size:18px"> 
            Интерфейс реализован для проверки работы модели прогнозирования объема закупки электроэнергии.  
            В поле Загрузить файл можно указать путь к файлу с данными.  
            Далее нужно выбрать Этап соревнований и нажать кнопку Run.  
            Результатом работы будет Датафрейм с датой и предсказанием на эту дату. 
            Также можно скачать файл predictions.csv с прогнозом
            </span> <center>
            """
        )
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(file_types=['.csv'], label='Загрузить файл', height=165)
            option_input = gr.Dropdown([
                'Этап 1. Предсказание на сутки.',
                'Этап 2. Предсказание на час.'
            ], label='Этап хакатона')
            btn = gr.Button("Run")
        with gr.Column(scale=2):
            df_output = gr.Dataframe(label='Predictions',
                                     col_count=2,
                                     max_rows=3,
                                     headers=['datetime', 'predict'],
                                     height=200,
                                     show_label=True)
            with gr.Row():
                with gr.Column(scale=2):
                    metrics_output = gr.Textbox(label='Метрики')
                with gr.Column(scale=1):
                    file_output = gr.File(label="Сохранить файл")
    with gr.Row():
        plot_output = gr.Plot(min_width=600)

    btn.click(fn=get_predictions, inputs=[file_input, option_input], outputs=[df_output, file_output, plot_output, metrics_output])

if __name__ == "__main__":
    iface.launch(show_api=False, server_name="95.140.157.138", server_port=7878)
