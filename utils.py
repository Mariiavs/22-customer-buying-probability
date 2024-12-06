#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import random

import seaborn as sns
import matplotlib.pyplot as plt

import io
from IPython.display import HTML

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, auc,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
)

sns.set_style('whitegrid')


# In[47]:


def display_error(error='Неизвестная ошибка') -> None:

    error_message = f"<p style='color:red;'>{error}!</b></p>"
    display(HTML(error_message))

    return None


def display_title(title='Заголовок', size=16, is_return=False) -> None:

    title = f"<br><p style='font-weight:bold; font-size: {size}px;'>{title}</p>"

    if is_return:
        return title

    else:
        display(HTML(title))

        return None


def display_info(df: pd.DataFrame, name: str, path: str, is_one=True):

    if is_one:
        
        display_title(name, size=20)
        print(f'Файл по пути {path} был успешно загружен.')
    
        display_title('Общие сведения:')
        display(df.info())
    
        display_title('Фрагмент датафрейма:')
        display(df.sample(5))

        return None

    else:

        content = display_title(name, size=20, is_return=True)
        content += f'<p>Файл по пути {path} был успешно загружен.</p>'
        content += display_title('Общие сведения:', is_return=True)

        buffer = io.StringIO()
        df.info(buf=buffer)
        info = buffer.getvalue()[1:].replace('\n', '<br>')

        content += f'<pre>{info}</pre>'
        content += display_title('Фрагмент датафрейма:', is_return=True)
        content += df.sample(5).to_html()

        return content


def display_tabs(tabs: list, tab_contents: list) -> None:
    
    id = f'tabs-group-{random.randint(100, 1000)}'
    tab_html = f"""
        <div class='{id}' style='margin: 20px;'>
            <ul class="tabs">
        """ 
    
    # Генерация ссылок для вкладок
    tab_html += ''.join(f'<li><a href="#{id}-tab-{i}" class="{"active-tab" if i == 0 else ""}">{tab}</a></li>'
                        for i, tab in enumerate(tabs)) + '</ul>'
    
    # Генерация содержимого вкладок
    tab_html += ''.join(
        f'<div id="{id}-tab-{i}" class="tab-content{" active" if i == 0 else ""}">{content}</div>'
        for i, (tab, content) in enumerate(zip(tabs, tab_contents))
    )
    
    tab_html += """
        </div>
        <style>
        .tabs { margin: 0; padding: 0; overflow: hidden; }
        .tabs li { list-style: none; float: left; margin: 0; padding: 0; }
        .tabs li a { display: block; padding: 10px; text-decoration: none; background: #E1F5FC; color: #000; }
        .tabs li a.active-tab { background: #C0E9F9; text-decoration: none; font-weight: bold; color: #333; }
        .tabs a:hover { background: #C0E9F9; }
        .tab-content { display: none; padding: 0 10px 10px 10px; }
        .tab-content.active { display: block; }
        </style>
        <script>
        // обработка кликов и добавление смещения
        document.querySelectorAll('.""" + str(id) + """ .tabs a').forEach(link => {
            link.addEventListener('click', function(event) {
                event.preventDefault();
        
                // Найти родительскую группу вкладок
                const parentGroup = this.closest('.""" + str(id) + """');
        
                // Сбросить активные классы в текущей группе
                parentGroup.querySelectorAll('.tabs a').forEach(tab => tab.classList.remove('active-tab'));
                parentGroup.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        
                // Активировать текущую вкладку и её содержимое
                this.classList.add('active-tab');
                parentGroup.querySelector(this.getAttribute('href')).classList.add('active');
            });
        });

        </script>
    """
    
    display(HTML(tab_html))

    return None


    
def load_csv(name: str, parse_dates=['date'], is_one=True) -> pd.DataFrame:
    
    path = f'datasets/{name}.csv'
    
    try:
        df = pd.read_csv(path, parse_dates=parse_dates)
        
    except:
        display_error(f'Ошибка чтения файла <b>{name}</b>')
        
        return None
        
    else:
        di = display_info(df, name, path, is_one)

        if is_one:
            return df
        else:
            return df, di
            

def load_files(name_list: list, dates_list: list) -> None:

    dfs = []
    tabs = []
    tab_contents = []

    for name, dates in zip(name_list, dates_list):
        if dates == []:
            dates = ['date']
        
        df, content = load_csv(name, parse_dates=dates, is_one=False)

        dfs.append(df)
        tabs.append(name)
        tab_contents.append(content)
    
    display_tabs(tabs, tab_contents)

    return dfs




# In[49]:


def handle_exceptions(func, exception=None):
    try:
        return func()
        
    except Exception as e:
        return exception


def parse_list(x):
    try:
        lst = list(map(int, x[2:-2].split("', '")))

        if len(lst) == 1:
            lst.append(-1)

        return lst
            
    except Exception:
        return [-1, -1]


# In[51]:


# обработка датафреймов
def process_dfs(names: list, dfs: list, func) -> None:

    tab_contents = []
    
    for name, df in zip(names, dfs):

        content = display_title(name, is_return=True)
        content += func(df, is_one=False)

        tab_contents.append(content)
        
    display_tabs(names, tab_contents)

    return None


# In[53]:


def is_na(data:pd.DataFrame, is_one=True):

    d = data.isna().sum()[data.isna().sum()>0].reset_index()
    d['доля, %'] = round(data.isna().mean()[data.isna().sum()>0]*100, 2).reset_index()[0]
    d = d.rename(columns={'index': 'столбец', 0: 'количество'})

    if is_one:
        return display_title('В таблице нет пропусков', size=14) if d.shape[0] == 0 else d
    else:
        return display_title('В таблице нет пропусков', size=14, is_return=True) if d.shape[0] == 0 else d.to_html()


def is_type(column: pd.Series, type=str, is_all=True) -> bool:

    types = column.apply(lambda x: isinstance(x, type))
    return types.all() if is_all else types.any()


def reshape_list(x: list, ls: list) -> None:
    ls.extend(x)
    return None


def unique(data: pd.DataFrame, columns=[], is_one=True):

    if len(columns) == 0:
        columns = data.select_dtypes(include='object').columns

    if len(columns) == 0:
        if is_one:
            display_title('В таблице нет категориальных признаков', size=14)
            return None    
        else:
            return display_title('В таблице нет категориальных признаков', size=14, is_return=True)

    count = []
    dfs = []

    for column in columns:

        if is_type(data[column]) or is_type(data[column], type=int):
            df = data[column].value_counts().reset_index()

        elif is_type(data[column], type=list):

            lst = []
            data[column].apply(lambda x: reshape_list(x, lst))

            df = pd.DataFrame(lst, columns=[column]).value_counts().reset_index()

        else:
            continue

        if df.shape[0] > 15:
            
            first_10 = df.head(10).copy()
            first_10.loc[first_10.shape[0]] = ['...', '...']
            last_5 = df.tail()
 
            result = pd.concat([first_10, last_5], ignore_index=True).drop_duplicates().reset_index(drop=True)
        
        else:
            result = df
        
        count.append([column, len(df[column].unique())])
        dfs.append(result)
        dfs.append(pd.DataFrame(['|' for i in range(result.shape[0])], columns=['|']))
    
    res = pd.concat(dfs, axis=1).fillna('')
    
    if is_one:
    
        display_title('Уникальные значения в категориальных столбцах:', size=14)
        display(pd.DataFrame(count, columns=['столбец', 'уникальные']))
        display(res)
    
        return res
    
    else:
        
        content = display_title('Уникальные значения в категориальных столбцах:', size=14, is_return=True)
        content += (pd.DataFrame(count, columns=['столбец', 'уникальные'])).to_html()
        content += res.to_html()
    
        return content


# In[55]:


def is_duplicates(data: pd.DataFrame, is_one=True) -> pd.DataFrame:
    
    duplicates = data.duplicated()
    part = round(duplicates.mean()*100, 2)

    if is_one:
        
        display_title(f'В таблице {part}% ({duplicates.sum()}) дубликатов', size=14)

        if part > 0:
            display(data[duplicates].head())
            
        return None
        

    else:
        content = display_title(f'В таблице {part}% ({duplicates.sum()}) дубликатов', size=14, is_return=True)

        if part > 0:
            content += data[duplicates].head().to_html()
            
        return content
    


# In[57]:


# EDA
def run_eda(data: pd.DataFrame, numeric_columns=None, categorical_columns=None, time_columns=None, id_columns=None, is_set=True):
    
    # исключение идентификаторов
    if id_columns:
        data = data.drop(columns=id_columns)

    if is_set:
        # автоматическое определение типов
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_columns is None:
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if time_columns is None:
            time_columns = data.select_dtypes(include=['datetime']).columns.tolist()

    # удаление пересечений между группами
    numeric_columns = [col for col in numeric_columns if col not in categorical_columns + time_columns]
    categorical_columns = [col for col in categorical_columns if col not in time_columns]

    # 1. анализ количественных признаков
    numeric_analysis_html = analyze_numeric(data, numeric_columns)
    
    # 2. анализ категориальных признаков
    categorical_analysis_html = analyze_categorical(data, categorical_columns)
    
    # 3. анализ временных признаков
    time_analysis_html = analyze_time(data, time_columns)
    
    # настройка отображения
    tabs = ['Количественные признаки', 'Категориальные признаки', 'Временные признаки']
    contents = [numeric_analysis_html, categorical_analysis_html, time_analysis_html]
    
    display_tabs(tabs, contents)


# анализ количественных признаков
def analyze_numeric(data, numeric_columns):
    
    if not numeric_columns:
        return '<p>Нет количественных признаков для анализа.</p>'

    stats = data[numeric_columns].describe().to_html()
    boxplots = generate_numeric_plots(data, numeric_columns, 'Гистограммы и диаграммы размаха')

    result = display_title('Сводная статистика', is_return=True) + stats
    result += display_title('Графики', is_return=True) + boxplots
    
    return result


def generate_category_plots(data, categorical_columns, top_n=4):
    
    plots_html = ""
    
    for col in categorical_columns:
        # Обработка данных для топ-N категорий
        value_counts = data[col].value_counts()
        
        if len(value_counts) > top_n:
            top_categories = value_counts.head(top_n)
            other_count = value_counts.iloc[top_n:].sum()
            plot_data = pd.DataFrame({
                col: list(top_categories.index) + ['Other'],
                'count': list(top_categories.values) + [other_count]
            })
        else:
            plot_data = value_counts.reset_index()
            plot_data.columns = [col, 'count']
            
        plot_data[col] = plot_data[col].astype(str)
        
        # Построение countplot
        fig, ax = plt.subplots(figsize=(6, 2))
        sns.barplot(data=plot_data, y=col, x='count', hue=col, ax=ax, palette='viridis', legend=False)
        ax.set_title(f'Распределение категорий для {col}')
        ax.set_xlabel('Количество')
        ax.set_ylabel('Категория')
        
        # Сохранение графика в HTML
        plots_html += save_plot_to_html(fig)
        plt.close(fig)
    return plots_html
    

# анализ категориальных признаков
def analyze_categorical(data, categorical_columns):
    
    if not categorical_columns:
        return '<p>Нет категориальных признаков для анализа.</p>'

    result = unique(data, columns=categorical_columns, is_one=False)
    result += generate_category_plots(data, categorical_columns)

    return result


# анализ временных признаков
def analyze_time(data, time_columns):
    
    if not time_columns:
        return '<p>Нет временных признаков для анализа.</p>'

    result = display_title('Анализ временных признаков', is_return=True)
    result += data[time_columns].describe().to_html()
    
    for col in time_columns:
        # result += data[col].describe().to_html()
        result += generate_time_plots(data, col)
        
    return result


# графики для количественных признаков
def generate_numeric_plots(data, columns, title='Распределения'):
    
    plots_html = ''
    for col in columns:
       
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(5, 1)

        # диаграмма размаха
        ax_box = fig.add_subplot(gs[:1, 0])
        sns.boxplot(x=data[col], ax=ax_box, color='#FF9999')
        ax_box.set_title(f"Распределение признака {col}")
        ax_box.set_xlabel('')
        ax_box.set_ylabel('')

        # гистограмма
        ax_hist = fig.add_subplot(gs[1:, 0])
        sns.histplot(data[col], kde=True, ax=ax_hist, color='#7F7FD5')
        ax_hist.set_ylabel('Количество')

        fig.tight_layout()
        plots_html += save_plot_to_html(fig)
        plt.close(fig)
        
    return plots_html


# временные графики
def generate_time_plots(data, column):
    
    fig, ax = plt.subplots(figsize=(10, 5))
    data[column].value_counts().sort_index().plot(ax=ax)
    ax.set_title(f'Тренд: {column}')
    
    plot_html = save_plot_to_html(fig)
    plt.close(fig)
    
    return plot_html


# сохранение графика в HTML
def save_plot_to_html(fig, width=100):
    
    from io import BytesIO
    import base64

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    return f'<img src="data:image/png;base64,{image_base64}" style="width: {width}%;"/>'



# In[59]:


# функция построения матрицы корреляции
def matrix(data: pd.DataFrame, size=(14, 11), title='Матрица корреляци\n', xlabel='Признаки', ylabel='Признаки', fmt='.3f'):
    
    fig, ax = plt.subplots(figsize=size)
    
    sns.heatmap(data, annot=True, fmt=fmt, center=0, cmap= 'PuOr')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    return fig
    


# In[61]:


def evaluate_model(l, is_one=True):

    y_true, y_pred, y_pred_proba = l[0], l[1], l[2]
    # Метрики
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc_value = auc(fpr, tpr)

    # F1
    f1_values = []
    thresholds_f1 = [i * 0.01 for i in range(100)]
    for threshold in thresholds_f1:
        y_pred_f1 = (y_pred_proba[:, 1] >= threshold).astype(int)
        f1_values.append(f1_score(y_true, y_pred_f1))

    # Плоты
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label='ROC-кривая (%0.2f)' % roc_auc_value)
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('\nROC-кривая\n')
    ax1.legend(loc="lower right")

    # F1 curve
    ax2.plot(thresholds_f1, f1_values, color='b', lw=2)
    ax2.set_xlabel('Порог')
    ax2.set_ylabel('F1 мера')
    ax2.set_title('\nF1 мера vs Порог\n')

    
    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = matrix(cm, size=(7, 5), title='Матрица ошибок\n', xlabel='Предсказанные значения', ylabel='Истинные значения', fmt='.0f')
    
    if is_one:
        
        print(f"accuracy: {accuracy}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1: {f1}")
        print(f"roc_auc: {roc_auc}")

    else:

        plt.close(fig)
        plt.close(fig_cm)

        content = """<div style='display: flex; align-items: flex-start; margin-top: 40px;'>"""
        content += """<div style='flex: none; width: 50%;'>""" + save_plot_to_html(fig_cm) + "</div>"
        content += f"""<div style='margin-top: 30px; margin-left: 50px; flex: none; width: 30%;'>
            <font size=3><b>Метрики</b><br><br></font>
            <p><b>accuracy:</b> {accuracy} <br></p>
            <p><b>precision:</b> {precision}<br></p>
            <p><b>recall:</b> {recall}<br></p>
            <p><b>f1:</b> {f1}<br></p>
            <p><b>roc_auc:</b> {roc_auc}<br></p>
        </div>""" 
        # content += """<div style='flex: none; width: 50%;'>""" + save_plot_to_html(fig_cm) + "</div>"
        
        content += '</div>' + save_plot_to_html(fig)

        return content

