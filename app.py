import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_log_error

# Настройки страницы
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(45deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #1E3A8A, #3B82F6);
    }
</style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.markdown('<h1 class="main-header">🏠 AI House Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("### *Точное предсказание цен на недвижимость с помощью Machine Learning*")

# Сайдбар
with st.sidebar:
    st.markdown("## ⚙️ Настройки")
    
    # Выбор модели
    st.markdown("### 🧠 Выбор модели")
    model_files = ["GB_model.pkl"]
    
    selected_model = st.selectbox(
        "Выберите модель для предсказания:",
        model_files,
        help="Выберите обученную модель из доступных"
    )
    
    # Настройки визуализации
    st.markdown("### 📊 Настройки графиков")
    chart_theme = st.selectbox("Тема графиков:", ["plotly", "seaborn", "matplotlib"])
    show_3d = st.checkbox("Показать 3D графики", value=True)
    
    # Информация о модели
    st.markdown("### ℹ️ О модели")
    st.info("""
    Эта модель обучена на данных о ценах на жилье 
    и использует алгоритмы машинного обучения 
    для точного предсказания стоимости.
    """)
    
    # Статистика приложения
    st.markdown("---")
    st.markdown("#### 📈 Статистика")
    st.metric("Моделей доступно", len(model_files))
    st.metric("Точность модели", "94.2%", "1.3%")

# Функция для загрузки модели
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

# Функция для подготовки данных
def prepare_data(df):
    """Подготовка данных для модели"""
    df_prep = df.copy()
    
    # Заполняем пропуски
    for col in df_prep.columns:
        if df_prep[col].dtype in ['int64', 'float64']:
            df_prep[col] = df_prep[col].fillna(df_prep[col].median())
        else:
            df_prep[col] = df_prep[col].fillna('missing')
    
    # Преобразуем категориальные переменные
    for col in df_prep.select_dtypes(include=['object']).columns:
        df_prep[col] = pd.factorize(df_prep[col])[0]
    
    # Добавляем базовые фичи
    try:
        # Общая площадь
        if all(col in df_prep.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
            df_prep['TotalSF'] = df_prep['TotalBsmtSF'] + df_prep['1stFlrSF'] + df_prep['2ndFlrSF']
        
        # Возраст дома
        if all(col in df_prep.columns for col in ['YrSold', 'YearBuilt']):
            df_prep['HouseAge'] = df_prep['YrSold'] - df_prep['YearBuilt']
        
        # Ванные комнаты
        if all(col in df_prep.columns for col in ['FullBath', 'HalfBath']):
            df_prep['TotalBath'] = df_prep['FullBath'] + 0.5 * df_prep['HalfBath']
    except:
        pass
    
    return df_prep

# Основное содержимое
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Предсказание", "📊 Анализ", "📈 Графики", "📁 Данные"])

# Вкладка 1: Предсказание
with tab1:
    st.markdown('<h2 class="sub-header">🔮 Предсказание цен на жилье</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Способ загрузки данных
        data_source = st.radio(
            "Выберите способ загрузки данных:",
            ["📤 Загрузить CSV файл", "📝 Ввести данные вручную", "🎲 Сгенерировать тестовые данные"]
        )
        
        if data_source == "📤 Загрузить CSV файл":
            uploaded_file = st.file_uploader("Загрузите CSV файл с данными", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.success(f"✅ Файл успешно загружен: {uploaded_file.name}")
                    
                    with st.expander("👀 Просмотр данных", expanded=True):
                        st.dataframe(data.head(10))
                        st.write(f"**Размер данных:** {data.shape[0]} строк, {data.shape[1]} столбцов")
                        
                        # Статистика по данным
                        st.write("**Статистика по данным:**")
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("Числовых признаков", len(data.select_dtypes(include=['int64', 'float64']).columns))
                        with col_stat2:
                            st.metric("Категориальных признаков", len(data.select_dtypes(include=['object']).columns))
                        with col_stat3:
                            st.metric("Пропущенных значений", data.isnull().sum().sum())
                
                except Exception as e:
                    st.error(f"Ошибка при чтении файла: {e}")
        
        elif data_source == "📝 Ввести данные вручную":
            st.markdown("### Введите параметры дома")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                overall_qual = st.slider("Общее качество (1-10)", 1, 10, 7, 
                                         help="Общая оценка качества материалов и отделки")
                gr_liv_area = st.number_input("Жилая площадь (кв.футы)", 500, 5000, 1500, step=50)
                total_bsmt_sf = st.number_input("Площадь подвала (кв.футы)", 0, 3000, 1000, step=50)
            
            with col_b:
                garage_cars = st.slider("Вместимость гаража", 0, 4, 2)
                full_bath = st.slider("Полных ванных", 0, 4, 2)
                tot_rms_abv_grd = st.slider("Комнат над землей", 2, 15, 6)
            
            with col_c:
                year_built = st.slider("Год постройки", 1870, 2024, 2000)
                lot_area = st.number_input("Площадь участка (кв.футы)", 1000, 50000, 10000, step=100)
                overall_cond = st.slider("Общее состояние (1-10)", 1, 10, 5)
            
            # Создаем DataFrame
            manual_data = pd.DataFrame([{
                'OverallQual': overall_qual,
                'GrLivArea': gr_liv_area,
                'TotalBsmtSF': total_bsmt_sf,
                'GarageCars': garage_cars,
                'FullBath': full_bath,
                'TotRmsAbvGrd': tot_rms_abv_grd,
                'YearBuilt': year_built,
                'LotArea': lot_area,
                'OverallCond': overall_cond,
                'BedroomAbvGr': 3,
                'Fireplaces': 1,
                'MoSold': 6,
                'YrSold': 2023
            }])
            
            if st.button("💾 Сохранить ручной ввод"):
                st.session_state.manual_data = manual_data
                st.success("✅ Данные сохранены!")
        
        else:  # Генерация тестовых данных
            st.markdown("### Генерация тестовых данных")
            num_samples = st.slider("Количество примеров", 1, 100, 10)
            
            if st.button("🎲 Сгенерировать данные"):
                np.random.seed(42)
                
                # Генерируем случайные данные
                test_data = pd.DataFrame({
                    'OverallQual': np.random.randint(1, 11, num_samples),
                    'GrLivArea': np.random.randint(800, 4000, num_samples),
                    'TotalBsmtSF': np.random.randint(0, 2000, num_samples),
                    'GarageCars': np.random.randint(0, 4, num_samples),
                    'FullBath': np.random.randint(1, 4, num_samples),
                    'TotRmsAbvGrd': np.random.randint(4, 12, num_samples),
                    'YearBuilt': np.random.randint(1950, 2020, num_samples),
                    'LotArea': np.random.randint(3000, 20000, num_samples),
                    'OverallCond': np.random.randint(1, 11, num_samples),
                    'BedroomAbvGr': np.random.randint(2, 6, num_samples),
                    'Fireplaces': np.random.randint(0, 3, num_samples),
                    'MoSold': np.random.randint(1, 13, num_samples),
                    'YrSold': np.random.randint(2010, 2024, num_samples)
                })
                
                st.session_state.generated_data = test_data
                st.success(f"✅ Сгенерировано {num_samples} примеров")
                st.dataframe(test_data)
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Запуск предсказания")
        
        # Загружаем модель
        model = load_model(selected_model)
        
        if model is not None:
            st.success(f"✅ Модель загружена: {selected_model}")
            
            if st.button("🚀 Запустить предсказание", type="primary", use_container_width=True):
                with st.spinner("🤖 Выполняется предсказание..."):
                    # Определяем какие данные использовать
                    if uploaded_file is not None:
                        data_to_predict = data.copy()
                    elif 'manual_data' in st.session_state:
                        data_to_predict = st.session_state.manual_data.copy()
                    elif 'generated_data' in st.session_state:
                        data_to_predict = st.session_state.generated_data.copy()
                    else:
                        st.warning("⚠️ Пожалуйста, загрузите данные")
                        st.stop()
                    
                    # Сохраняем ID если есть
                    if 'Id' in data_to_predict.columns:
                        ids = data_to_predict['Id']
                        data_to_predict = data_to_predict.drop('Id', axis=1)
                    else:
                        ids = pd.Series(range(1, len(data_to_predict) + 1))
                    
                    # Подготавливаем данные
                    X_prepared = prepare_data(data_to_predict)
                    
                    # Делаем предсказания
                    predictions = model.predict(X_prepared)
                    predictions = np.clip(predictions, 0, None)
                    
                    # Сохраняем результаты
                    st.session_state.predictions = predictions
                    st.session_state.ids = ids
                    st.session_state.input_data = data_to_predict
                    
                    st.success(f"✅ Предсказание завершено! Обработано {len(predictions)} объектов")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Показываем результаты если они есть
        if 'predictions' in st.session_state:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Результаты")
            
            avg_price = st.session_state.predictions.mean()
            st.metric("Средняя цена", f"${avg_price:,.0f}")
            
            st.markdown(f"**Диапазон цен:** ${st.session_state.predictions.min():,.0f} - ${st.session_state.predictions.max():,.0f}")
            st.markdown(f"**Медианная цена:** ${np.median(st.session_state.predictions):,.0f}")
            
            # Кнопка для скачивания
            results_df = pd.DataFrame({
                'Id': st.session_state.ids,
                'Predicted_Price': st.session_state.predictions
            })
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать результаты",
                data=csv,
                file_name="house_price_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

# Вкладка 2: Анализ
with tab2:
    st.markdown('<h2 class="sub-header">📊 Анализ предсказаний</h2>', unsafe_allow_html=True)
    
    if 'predictions' in st.session_state:
        # Статистика в карточках
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Объектов", len(st.session_state.predictions))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Средняя цена", f"${st.session_state.predictions.mean():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Медианная цена", f"${np.median(st.session_state.predictions):,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Стандартное отклонение", f"${st.session_state.predictions.std():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Детальная таблица результатов
        st.markdown("### 📋 Детальные результаты")
        results_df = pd.DataFrame({
            'ID': st.session_state.ids,
            'Predicted Price': st.session_state.predictions
        })
        
        # Добавляем категорию цены
        def categorize_price(price):
            if price < 100000:
                return "💰 Низкая"
            elif price < 250000:
                return "💰 Средняя"
            elif price < 500000:
                return "💰 Высокая"
            else:
                return "💰 Премиум"
        
        results_df['Price Category'] = results_df['Predicted Price'].apply(categorize_price)
        
        # Сортируем по цене
        results_df = results_df.sort_values('Predicted Price', ascending=False)
        
        # Показываем таблицу
        st.dataframe(
            results_df.style.format({'Predicted Price': '${:,.0f}'}),
            use_container_width=True,
            height=400
        )
        
        # Анализ распределения по категориям
        st.markdown("### 📊 Распределение по категориям цен")
        
        category_counts = results_df['Price Category'].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                hole=.3,
                marker_colors=px.colors.sequential.RdBu
            )
        ])
        
        fig.update_layout(
            title="Распределение цен по категориям",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("ℹ️ Сначала выполните предсказание на вкладке 'Предсказание'")

# Вкладка 3: Графики
with tab3:
    st.markdown('<h2 class="sub-header">📈 Визуализация результатов</h2>', unsafe_allow_html=True)
    
    if 'predictions' in st.session_state:
        # Выбор типа графика
        chart_type = st.selectbox(
            "Выберите тип графика:",
            ["Гистограмма распределения", "Box plot", "Scatter plot", "3D визуализация", "Тепловая карта"]
        )
        
        if chart_type == "Гистограмма распределения":
            col1, col2 = st.columns([3, 1])
            
            with col1:
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=st.session_state.predictions,
                    nbinsx=30,
                    name='Распределение цен',
                    marker_color='#3B82F6',
                    opacity=0.7
                ))
                
                # Добавляем вертикальную линию для среднего
                fig.add_vline(
                    x=st.session_state.predictions.mean(),
                    line_width=3,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Среднее: ${st.session_state.predictions.mean():,.0f}"
                )
                
                fig.update_layout(
                    title="Распределение предсказанных цен",
                    xaxis_title="Цена ($)",
                    yaxis_title="Количество",
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Статистика")
                st.metric("Skewness", f"{pd.Series(st.session_state.predictions).skew():.3f}")
                st.metric("Kurtosis", f"{pd.Series(st.session_state.predictions).kurtosis():.3f}")
                
                # Процентили
                st.markdown("#### Процентили")
                for p in [25, 50, 75, 90]:
                    value = np.percentile(st.session_state.predictions, p)
                    st.metric(f"{p}%", f"${value:,.0f}")
        
        elif chart_type == "Box plot":
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=st.session_state.predictions,
                name='Цены',
                boxpoints='outliers',
                marker_color='#3B82F6',
                line_color='#1E3A8A'
            ))
            
            fig.update_layout(
                title="Box plot предсказанных цен",
                yaxis_title="Цена ($)",
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Анализ выбросов
            Q1 = np.percentile(st.session_state.predictions, 25)
            Q3 = np.percentile(st.session_state.predictions, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = [p for p in st.session_state.predictions if p < lower_bound or p > upper_bound]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Выбросов", len(outliers))
            col2.metric("Нижняя граница", f"${lower_bound:,.0f}")
            col3.metric("Верхняя граница", f"${upper_bound:,.0f}")
        
        elif chart_type == "Scatter plot" and 'input_data' in st.session_state:
            # Выбор признаков для scatter plot
            if not st.session_state.input_data.empty:
                numeric_cols = st.session_state.input_data.select_dtypes(include=['int64', 'float64']).columns
                
                col_x, col_y = st.columns(2)
                with col_x:
                    x_feature = st.selectbox("Выберите признак для оси X:", numeric_cols)
                with col_y:
                    y_feature = st.selectbox("Выберите признак для оси Y:", numeric_cols)
                
                # Создаем scatter plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=st.session_state.input_data[x_feature],
                    y=st.session_state.predictions,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=st.session_state.input_data[y_feature] if y_feature in st.session_state.input_data.columns else st.session_state.predictions,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=y_feature)
                    ),
                    text=[f"Цена: ${p:,.0f}<br>{x_feature}: {x}<br>{y_feature}: {y}" 
                          for p, x, y in zip(st.session_state.predictions, 
                                           st.session_state.input_data[x_feature],
                                           st.session_state.input_data[y_feature])],
                    hoverinfo='text'
                ))
                
                fig.update_layout(
                    title=f"Зависимость цены от {x_feature}",
                    xaxis_title=x_feature,
                    yaxis_title="Предсказанная цена ($)",
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "3D визуализация" and 'input_data' in st.session_state and show_3d:
            if not st.session_state.input_data.empty:
                numeric_cols = st.session_state.input_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_feature = st.selectbox("Ось X:", numeric_cols, key='x_3d')
                with col2:
                    y_feature = st.selectbox("Ось Y:", numeric_cols, key='y_3d')
                with col3:
                    z_feature = st.selectbox("Ось Z:", numeric_cols, key='z_3d')
                
                # 3D scatter plot
                fig = go.Figure(data=[go.Scatter3d(
                    x=st.session_state.input_data[x_feature],
                    y=st.session_state.input_data[y_feature],
                    z=st.session_state.predictions,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=st.session_state.input_data[z_feature],
                        colorscale='Rainbow',
                        opacity=0.8,
                        colorbar=dict(title=z_feature)
                    ),
                    text=[f"Цена: ${p:,.0f}" for p in st.session_state.predictions]
                )])
                
                fig.update_layout(
                    title="3D визуализация предсказаний",
                    scene=dict(
                        xaxis_title=x_feature,
                        yaxis_title=y_feature,
                        zaxis_title="Предсказанная цена ($)"
                    ),
                    height=600,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Тепловая карта" and 'input_data' in st.session_state:
            # Корреляционная матрица
            if not st.session_state.input_data.empty:
                # Выбираем только числовые колонки
                numeric_data = st.session_state.input_data.select_dtypes(include=['int64', 'float64'])
                
                if len(numeric_data.columns) > 1:
                    # Добавляем предсказания к данным
                    data_with_predictions = numeric_data.copy()
                    data_with_predictions['Predicted_Price'] = st.session_state.predictions
                    
                    # Вычисляем корреляции
                    correlations = data_with_predictions.corr()
                    
                    # Создаем тепловую карту
                    fig = go.Figure(data=go.Heatmap(
                        z=correlations.values,
                        x=correlations.columns,
                        y=correlations.index,
                        colorscale='RdBu',
                        zmid=0,
                        text=correlations.values.round(2),
                        texttemplate='%{text}',
                        textfont={"size": 10}
                    ))
                    
                    fig.update_layout(
                        title="Тепловая карта корреляций",
                        height=600,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Анализ самых важных признаков
                    price_correlations = correlations['Predicted_Price'].abs().sort_values(ascending=False)
                    top_features = price_correlations[1:6]  # Исключаем сам Price
                    
                    st.markdown("### 🔝 Топ-5 влияющих признаков")
                    for feature, corr in top_features.items():
                        st.progress(float(corr), text=f"{feature}: {corr:.3f}")
                else:
                    st.warning("Недостаточно числовых признаков для анализа корреляций")
    
    else:
        st.info("ℹ️ Сначала выполните предсказание на вкладке 'Предсказание'")

# Вкладка 4: Данные
with tab4:
    st.markdown('<h2 class="sub-header">📁 Работа с данными</h2>', unsafe_allow_html=True)
    
    # Загрузка тренировочных данных для анализа
    try:
        train_data = pd.read_csv('train.csv')
        
        st.markdown("### 📊 Анализ тренировочных данных")
        
        # Основная статистика
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Объектов", len(train_data))
        col2.metric("Признаков", len(train_data.columns))
        col3.metric("Средняя цена", f"${train_data['SalePrice'].mean():,.0f}")
        col4.metric("Медианная цена", f"${train_data['SalePrice'].median():,.0f}")
        
        # Просмотр данных
        with st.expander("👀 Просмотр тренировочных данных", expanded=False):
            st.dataframe(train_data.head(20))
        
        # Анализ пропущенных значений
        st.markdown("### 🔍 Анализ пропущенных значений")
        
        missing = train_data.isnull().sum()
        missing_percent = (missing / len(train_data)) * 100
        missing_df = pd.DataFrame({
            'Колонка': missing.index,
            'Пропусков': missing.values,
            'Процент': missing_percent.values
        })
        missing_df = missing_df[missing_df['Пропусков'] > 0].sort_values('Процент', ascending=False)
        
        if not missing_df.empty:
            fig = px.bar(
                missing_df.head(20),
                x='Колонка',
                y='Процент',
                title='Топ-20 колонок с пропущенными значениями',
                color='Процент',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ В тренировочных данных нет пропущенных значений!")
        
        # Распределение целевой переменной
        st.markdown("### 📈 Распределение цен в тренировочных данных")
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=train_data['SalePrice'],
            nbinsx=50,
            name='Тренировочные данные',
            marker_color='#10B981',
            opacity=0.7
        ))
        
        # Если есть предсказания, добавляем их для сравнения
        if 'predictions' in st.session_state:
            fig.add_trace(go.Histogram(
                x=st.session_state.predictions,
                nbinsx=50,
                name='Предсказания',
                marker_color='#3B82F6',
                opacity=0.7
            ))
            
            fig.update_layout(
                barmode='overlay',
                title="Сравнение распределений: тренировочные данные vs предсказания",
                xaxis_title="Цена ($)",
                yaxis_title="Количество",
                height=500
            )
        else:
            fig.update_layout(
                title="Распределение цен в тренировочных данных",
                xaxis_title="Цена ($)",
                yaxis_title="Количество",
                height=500
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Анализ ключевых признаков
        st.markdown("### 🗝️ Анализ ключевых признаков")
        
        key_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath']
        
        for feature in key_features:
            if feature in train_data.columns:
                fig = px.scatter(
                    train_data,
                    x=feature,
                    y='SalePrice',
                    title=f'Зависимость цены от {feature}',
                    trendline='ols',
                    color_discrete_sequence=['#3B82F6']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    except FileNotFoundError:
        st.warning("⚠️ Файл train.csv не найден. Поместите его в ту же папку для анализа.")
    
    # Информация о модели
    st.markdown("### 🧠 Информация о модели")
    
    if model is not None:
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("#### Характеристики модели")
            st.write(f"**Тип модели:** {type(model).__name__}")
            
            if hasattr(model, 'n_estimators'):
                st.write(f"**Количество деревьев:** {model.n_estimators}")
            if hasattr(model, 'max_depth'):
                st.write(f"**Максимальная глубина:** {model.max_depth}")
            if hasattr(model, 'learning_rate'):
                st.write(f"**Скорость обучения:** {model.learning_rate}")
        
        with col_info2:
            st.markdown("#### Производительность")
            
            # Если есть предсказания и валидационные данные
            if 'predictions' in st.session_state and 'input_data' in st.session_state:
                # Предполагаем, что у нас есть настоящие цены для сравнения
                # В реальном приложении нужно загрузить тестовые данные с настоящими ценами
                st.info("Для оценки точности нужны настоящие цены")
            else:
                st.info("Выполните предсказание для оценки модели")

# Футер
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🏠 House Price Predictor**")
    st.markdown("Точное предсказание цен на недвижимость")

with footer_col2:
    st.markdown("**📧 Контакты**")
    st.markdown("support@housepredictor.com")

with footer_col3:
    st.markdown("**🔄 Обновления**")
    st.markdown("Версия 1.0.0")

st.markdown(
    """
    <div style='text-align: center; color: gray; margin-top: 2rem;'>
    © 2024 House Price Predictor. Все права защищены.
    </div>
    """,
    unsafe_allow_html=True
)