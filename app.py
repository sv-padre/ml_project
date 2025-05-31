### OVERALL CODE
#либы
#база
#!pip install streamlit
#!pip install ydata-profiling

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
import inspect
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Keras/TensorFlow для перцептрона
import keras
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier

# Настройка страницы Streamlit
st.set_page_config(page_title="HSE | ТИИПА", layout="wide")

# Инициализация session_state
def initialize_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.backup_df = None
        st.session_state.X_train = None
        st.session_state.X_test = None
        st.session_state.y_train = None
        st.session_state.y_test = None
        st.session_state.train_ids = None
        st.session_state.test_ids = None
        st.session_state.models = {}
        st.session_state.scaler = None
        st.session_state.onehot_encoder = None
        st.session_state.label_encoders = {}
        st.session_state.target_col = None
        st.session_state.id_cols = []
        st.session_state.selected_features = []
        st.session_state.norm_cols = []
        st.session_state.log_cols = []
        st.session_state.dummy_cols = []
        st.session_state.balance_method = "Нет"
        st.session_state.method = "Заменить на медиану"
        st.session_state.sep_sign = ";"
        st.session_state.decimal_sign = ","
        st.session_state.ensemble_model = None
        st.session_state.preprocess_params = {}

initialize_session_state()

# Заголовок и информация
st.title("© HSE | ТИИПА")
st.markdown("**Fast-touch classification-analytical tool**")
st.markdown("---")

# Блок информации о команде
with st.expander("ℹ О команде разработчиков"):
    st.markdown("""
    **Команда проекта:**
    - Константин Ильященко — Team Leader
    - Вадим Казаков — Master of Machine Learning
    - Андрей Ширшов — Business Developer
    - Татьяна Черных — Designer
    """)

# Инструкция по использованию
with st.expander("Инструкция по использованию сервиса", expanded=True):
    st.markdown("""
    1. *Выберите нужную тему* анализа
    2. *Подготовьте данные* с переменной для предсказания и переменными для анализа
    3. *Загрузите данные* и выберите подходящие переменные
    4. *Запустите анализ* — выберите модели машинного обучения и доверьтесь нам. *Дайте алгоритму время для обработки.*
    5. *Изучите результаты* — метрики предсказания и визуализации.
    6. *Выберите* лучшую модель и *примените* её на совершенно новых данных.
    """)

st.markdown("---")

# Функция для загрузки данных
def load_data(uploaded_file):
    try:
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith('.csv'):
            sep_sign = st.selectbox(
                "Выберите разделитель",
                (";", ",", " ", "|"), index=0)
            st.session_state.sep_sign = sep_sign

            decimal_sign = st.selectbox(
                "Выберите отделитель дробной части",
                (".", ","), index=1)
            st.session_state.decimal_sign = decimal_sign

            df = pd.read_csv(uploaded_file, sep=sep_sign, decimal=decimal_sign)
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        
        st.session_state.df = df.copy()
        st.session_state.backup_df = df.copy()
        st.success("Данные успешно загружены!")
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        return None

# Функция для обработки пропущенных значений
def handle_missing_values(df, option):
    df_processed = df.copy()
    
    for col in df_processed.columns:
        if pd.api.types.is_numeric_dtype(df_processed[col]):
            if option == "Удалить строки с пропусками":
                df_processed = df_processed.dropna(subset=[col])
            elif option == "Заменить на ноль":
                df_processed[col] = df_processed[col].fillna(0)
            elif option == "Заменить на медиану":
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            elif option == "Заменить на среднее":
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            elif option == "Интерполяция":
                df_processed[col] = df_processed[col].interpolate()
        else:
            if option == "Удалить строки с пропусками":
                df_processed = df_processed.dropna(subset=[col])
            elif option == "Заменить на моду":
                if not df_processed[col].mode().empty:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                else:
                    df_processed[col] = df_processed[col].fillna("Unknown")
            elif option == "Создать новую категорию 'Unknown'":
                df_processed[col] = df_processed[col].fillna("Unknown")
            elif option == "Экстраполяция (последнее значение)":
                df_processed[col] = df_processed[col].fillna(method='ffill')
    
    return df_processed

# Функция для предобработки данных
def preprocess_data(data, target_col, id_cols, features, norm_cols, log_cols, dummy_cols,
                   balance_method, test_size, random_state):
    try:
        # Сохраняем параметры предобработки
        st.session_state.preprocess_params = {
            'target_col': target_col,
            'features': features,
            'norm_cols': norm_cols,
            'log_cols': log_cols,
            'dummy_cols': dummy_cols,
            'balance_method': balance_method
        }
        
        X = data[features].copy()
        y = data[target_col].copy()
        ids = data[id_cols].copy() if id_cols else None

        # Кодируем категориальный таргет
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.session_state.label_encoders['target'] = le

        # Разбиваем на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size/100,
            random_state=random_state,
            stratify=y if not pd.api.types.is_numeric_dtype(y) else None
        )

        # Сохраняем ID
        if ids is not None:
            train_ids = ids.loc[X_train.index]
            test_ids = ids.loc[X_test.index]
        else:
            train_ids, test_ids = None, None

        # 1. Логарифмирование
        for col in log_cols:
            if col in X_train.columns:
                if (X_train[col] <= 0).any():
                    st.warning(f"Колонка {col} содержит нули/отрицательные значения! Добавлена константа 1 перед логарифмированием.")
                    X_train[col] = np.log1p(X_train[col] + 1)
                    X_test[col] = np.log1p(X_test[col] + 1)
                else:
                    X_train[col] = np.log1p(X_train[col])
                    X_test[col] = np.log1p(X_test[col])

        # 2. One-hot кодирование
        if dummy_cols:
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            train_encoded = encoder.fit_transform(X_train[dummy_cols])
            test_encoded = encoder.transform(X_test[dummy_cols])

            encoded_cols = encoder.get_feature_names_out(dummy_cols)
            train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=X_train.index)
            test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=X_test.index)

            X_train = X_train.drop(columns=dummy_cols).join(train_encoded_df)
            X_test = X_test.drop(columns=dummy_cols).join(test_encoded_df)
            st.session_state.onehot_encoder = encoder

        # 3. Нормализация
        if norm_cols:
            scaler = StandardScaler()
            X_train[norm_cols] = scaler.fit_transform(X_train[norm_cols])
            X_test[norm_cols] = scaler.transform(X_test[norm_cols])
            st.session_state.scaler = scaler

        # 4. Балансировка классов
        if balance_method != "Нет" and not pd.api.types.is_numeric_dtype(y_train):
            if balance_method == "Random Oversampling":
                sampler = RandomOverSampler(random_state=random_state)
            elif balance_method == "SMOTE":
                sampler = SMOTE(random_state=random_state)
            elif balance_method == "Random Undersampling":
                sampler = RandomUnderSampler(random_state=random_state)
            
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            
            # Визуализация балансировки
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            pd.Series(y).value_counts().plot(kind='bar', ax=ax1, title='До балансировки')
            pd.Series(y_train).value_counts().plot(kind='bar', ax=ax2, title='После балансировки')
            st.pyplot(fig)

        return X_train, X_test, y_train, y_test, train_ids, test_ids
    
    except Exception as e:
        st.error(f"Ошибка при предобработке данных: {str(e)}")
        return None, None, None, None, None, None

# Функция для обработки новых данных
def process_new_data(new_df):
    try:
        params = st.session_state.preprocess_params
        if not params:
            raise ValueError("Сначала нужно подготовить тренировочные данные!")
        
        X_new = new_df[params['features']].copy()
        
        # 1. Логарифмирование
        for col in params['log_cols']:
            if col in X_new.columns:
                X_new[col] = np.log1p(X_new[col] + 1)
        
        # 2. One-hot кодирование
        if 'onehot_encoder' in st.session_state and params['dummy_cols']:
            encoded = st.session_state.onehot_encoder.transform(X_new[params['dummy_cols']])
            encoded_cols = st.session_state.onehot_encoder.get_feature_names_out(params['dummy_cols'])
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X_new.index)
            X_new = X_new.drop(params['dummy_cols'], axis=1).join(encoded_df)
        
        # 3. Нормализация
        if 'scaler' in st.session_state and params['norm_cols']:
            X_new[params['norm_cols']] = st.session_state.scaler.transform(X_new[params['norm_cols']])
        
        return X_new
    
    except Exception as e:
        st.error(f"Ошибка при обработке новых данных: {str(e)}")
        return None

# Основной интерфейс
st.title("Данные для анализа")
uploaded_file = st.file_uploader("Выберите файл (CSV или Excel)", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.dataframe(df.head())

col1, col2 = st.columns(2)
with col1:
    method = st.selectbox(
        "Метод обработки пропусков:",
        options=[
            "Удалить строки с пропусками",
            "Заменить на ноль",
            "Заменить на медиану",
            "Заменить на среднее",
            "Заменить на моду",
            "Создать новую категорию 'Unknown'",
            "Интерполяция",
            "Экстраполяция (последнее значение)"
        ],
        index=2,
        help="Выберите метод обработки пропущенных значений",
        key="missing_method_select"
    )
    st.session_state.method = method # Сохраняем выбранный метод

with col2:
    show_details = st.checkbox("Показать пояснения методов", value=True)

if show_details:
    with st.expander("📚 Пояснения методов обработки"):
        st.markdown("""
        - **Удалить строки с пропусками** - полное удаление строк, содержащих пропуски
        - **Заменить на ноль** - замена всех пропусков на 0 (для числовых данных)
        - **Заменить на медиану** - замена пропусков медианным значением (устойчиво к выбросам)
        - **Заменить на среднее** - замена пропусков средним значением (может искажаться выбросами)
        - **Заменить на моду** - замена пропусков наиболее частым значением (для категориальных данных)
        - **Создать новую категорию 'Unknown'** - для категориальных данных, сохраняет информацию о пропуске
        - **Интерполяция** - линейная интерполяция значений (для временных рядов)
        - **Экстраполяция** - заполнение последним известным значением (метод 'forward fill')
        """)

# Обработка данных
if st.checkbox("Обработать пропущенные значения", key="process_missing_checkbox"):
    with st.spinner("Обработка данных..."):
        df = handle_missing_values(df, method)
        st.session_state.df = df.copy() # Обновляем df в session_state
        st.success("Обработка завершена!")

        # Вывод результатов
        st.markdown("---")
        st.subheader("Результаты обработки")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Исходное количество строк", len(backup_df))
        with col2:
            st.metric("Количество строк после обработки", len(df))

# Показать обработанный датафрейм
st.dataframe(st.session_state.df)


# Визуальный анализ данных
if st.session_state.df is not None:
    st.title("Визуальный анализ данных")
    st.subheader("Описательная статистика")

    desc = st.session_state.df.describe().T
    desc['missing'] = st.session_state.df.isna().sum()
    desc['missing_percent'] = (desc['missing'] / len(st.session_state.df)).round(2)
    desc['dtype'] = st.session_state.df.dtypes
    desc['unique'] = st.session_state.df.nunique()

    st.dataframe(
        desc.style.format({
            'mean': '{:.2f}',
            'std': '{:.2f}',
            'min': '{:.2f}',
            '25%': '{:.2f}',
            '50%': '{:.2f}',
            '75%': '{:.2f}',
            'max': '{:.2f}',
            'missing_percent': '{:.0%}'
        }),
        use_container_width=True
    )
    

    # Графики распределения
    st.subheader("Распределение признаков")
    col1, col2 = st.columns([3, 1])
    with col1:
        numeric_cols_for_plot = st.session_state.df.select_dtypes(include=['number']).columns
        if len(numeric_cols_for_plot) > 0:
            selected_col = st.selectbox(
                "Выберите переменную для анализа:",
                options=numeric_cols_for_plot,
                index=0
            )
        else:
            st.warning("Нет числовых колонок для построения графиков распределения.")
            selected_col = None
    
    with col2:
        bins = 20 # Значение по умолчанию
        if chart_type == "Гистограмма" and selected_col:
            bins = st.slider(
                "Количество интервалов:",
                min_value=5,
                max_value=50,
                value=int(np.sqrt(len(st.session_state.df)) if len(st.session_state.df) > 0 else 20)
            )
    
    # Отображение выбранного графика
    if selected_col:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
    
        if chart_type == "Гистограмма":
            sns.histplot(st.session_state.df[selected_col], bins=bins, kde=True, ax=ax)
            ax.set_title(f'Распределение {selected_col}')
            ax.set_xlabel(selected_col)
            ax.set_ylabel('Частота')
        elif chart_type == "Ящик с усами (Boxplot)":
            sns.boxplot(x=st.session_state.df[selected_col], ax=ax)
            ax.set_title(f'Boxplot для {selected_col}')
        elif chart_type == "Плотность распределения":
            sns.kdeplot(st.session_state.df[selected_col], ax=ax, fill=True)
            ax.set_title(f'Плотность распределения {selected_col}')
            ax.set_xlabel(selected_col)
            ax.set_ylabel('Плотность')
    
        st.pyplot(fig, clear_figure=True)
    
    st.markdown("---")
    
    st.subheader("Scatter plot")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_axis = st.selectbox(
            "Ось X",
            options=st.session_state.df.columns,
            index=0
        )
    with col2:
        y_axis = st.selectbox(
            "Ось Y",
            options=st.session_state.df.columns,
            index=1 if len(st.session_state.df.columns) > 1 else 0
        )
    
    with col3:
        color_col = st.selectbox(
            "Цвет",
            options=["None"] + list(st.session_state.df.columns),
            index=0
        )
    with col4:
        size_col = st.selectbox(
            "Размер",
            options=["None"] + list(st.session_state.df.columns),
            index=0
        )
    
    if (color_col == "None") and (size_col == "None"):
        st.scatter_chart(st.session_state.df, x=x_axis, y=y_axis)
    elif (size_col == "None") and (color_col != "None"):
        st.scatter_chart(st.session_state.df, x=x_axis, y=y_axis, color=color_col)
    elif (size_col != "None") and (color_col == "None"):
        st.scatter_chart(st.session_state.df, x=x_axis, y=y_axis, size = size_col)
    elif (size_col != "None") and (color_col != "None"):
        st.scatter_chart(st.session_state.df, x=x_axis, y=y_axis, color = color_col, size = size_col)

# Подготовка данных для моделирования
if st.session_state.df is not None:
    st.title("Подготовка данных для моделирования")
    
    target_col = st.selectbox(
        "Выберите целевую переменную:",
        options=st.session_state.df.columns,
        index=0
    )
    st.session_state.target_col = target_col

    # Анализ по классам
    if st.session_state.target_col:
        st.subheader(f"Анализ по классам ({st.session_state.target_col})")
        class_stats = st.session_state.df.groupby(st.session_state.target_col).agg(['mean', 'count', 'nunique']).T
        st.dataframe(class_stats)
        
        fig, ax = plt.subplots()
        sns.countplot(x=st.session_state.target_col, data=st.session_state.df, ax=ax)
        st.pyplot(fig)

    id_cols = st.multiselect(
        "Выберите ID-столбцы (не будут использоваться в модели):",
        options=[col for col in st.session_state.df.columns if col != target_col]
    )
    st.session_state.id_cols = id_cols

    available_features = [col for col in st.session_state.df.columns if col != target_col and col not in id_cols]
    selected_features = st.multiselect(
        "Выберите признаки для модели:",
        options=available_features,
        default=available_features
    )
    st.session_state.selected_features = selected_features

    # Настройки предобработки
    with st.expander("Настройки предобработки"):
        numeric_cols = [col for col in selected_features if pd.api.types.is_numeric_dtype(st.session_state.df[col])]
        
        st.subheader("Нормализация")
        norm_cols = st.multiselect(
            "Признаки для нормализации:",
            options=numeric_cols
        )
        st.session_state.norm_cols = norm_cols

        st.subheader("Логарифмирование")
        log_cols = st.multiselect(
            "Признаки для логарифмирования:",
            options=numeric_cols
        )
        st.session_state.log_cols = log_cols

        st.subheader("Кодирование категорий")
        categorical_cols = [col for col in selected_features if not pd.api.types.is_numeric_dtype(st.session_state.df[col])]
        dummy_cols = st.multiselect(
            "Категориальные признаки для one-hot кодирования:",
            options=categorical_cols
        )
        st.session_state.dummy_cols = dummy_cols

        st.subheader("Балансировка классов")
        if not pd.api.types.is_numeric_dtype(st.session_state.df[target_col]):
            balance_method = st.selectbox(
                "Метод балансировки:",
                options=["Нет", "Random Oversampling", "SMOTE", "Random Undersampling"]
            )
            st.session_state.balance_method = balance_method
        else:
            st.warning("Балансировка доступна только для категориальных целевых переменных")

    # Параметры разбиения
    test_size = st.slider(
        "Размер тестовой выборки (%)",
        min_value=5,
        max_value=40,
        value=20,
        step=5
    )
    
    random_state = st.number_input(
        "Random state для воспроизводимости",
        min_value=0,
        value=42
    )

    if st.button("Подготовить данные"):
        with st.spinner("Обработка данных..."):
            X_train, X_test, y_train, y_test, train_ids, test_ids = preprocess_data(
                st.session_state.df,
                st.session_state.target_col,
                st.session_state.id_cols,
                st.session_state.selected_features,
                st.session_state.norm_cols,
                st.session_state.log_cols,
                st.session_state.dummy_cols,
                st.session_state.balance_method,
                test_size,
                random_state
            )
            
            if X_train is not None:
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.train_ids = train_ids
                st.session_state.test_ids = test_ids
                
                st.success("Данные успешно подготовлены!")
                
                # Показать результаты
                st.subheader("Результаты подготовки")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Обучающая выборка", f"{len(X_train)} строк")
                    st.write("Распределение классов:")
                    st.write(pd.Series(y_train).value_counts())
                    st.dataframe(pd.concat([train_ids, X_train, y_train], axis=1))
                with col2:
                    st.metric("Тестовая выборка", f"{len(X_test)} строк")
                    st.write("Распределение классов:")
                    st.write(pd.Series(y_test).value_counts())
                    st.dataframe(pd.concat([test_ids, X_test, y_test], axis=1))
            else:
                st.error("Ошибка при подготовке данных")
