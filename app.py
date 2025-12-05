import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px

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
    
    st.markdown("### ⚡ Быстрый режим")
    quick_mode = st.checkbox("Использовать быстрый режим", value=True,
                           help="Использовать только основные признаки для быстрого предсказания")

# Функция для загрузки модели
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

# Функция для подготовки данных (основные признаки)
def prepare_basic_features(df):
    """Подготовка только основных признаков"""
    df = df.copy()
    
    # Основные признаки, которые используются в большинстве моделей
    basic_features = [
        'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 
        'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
        'LotArea', 'OverallCond', 'BedroomAbvGr', 'Fireplaces',
        'GarageArea', 'MoSold', 'YrSold', '1stFlrSF', '2ndFlrSF'
    ]
    
    result = pd.DataFrame()
    
    for feature in basic_features:
        if feature in df.columns:
            result[feature] = df[feature]
        else:
            # Если признака нет, заполняем медианой или 0
            if feature in ['OverallQual', 'GarageCars', 'FullBath', 'TotRmsAbvGrd', 
                          'OverallCond', 'BedroomAbvGr', 'Fireplaces']:
                result[feature] = 0  # для категориальных/дискретных
            else:
                result[feature] = 0  # для числовых
    
    # Заполняем пропуски
    result = result.fillna(0)
    
    return result

# Функция для ручного ввода
def create_manual_input():
    """Создание данных для ручного ввода"""
    st.markdown("### 📝 Введите параметры дома")
    
    col1, col2 = st.columns(2)
    
    with col1:
        overall_qual = st.slider("Общее качество (1-10)", 1, 10, 7)
        gr_liv_area = st.number_input("Жилая площадь (кв.футы)", 500, 5000, 1500, step=50)
        total_bsmt_sf = st.number_input("Площадь подвала (кв.футы)", 0, 3000, 1000, step=50)
        garage_cars = st.slider("Вместимость гаража", 0, 4, 2)
        full_bath = st.slider("Полных ванных", 0, 4, 2)
    
    with col2:
        tot_rms_abv_grd = st.slider("Комнат над землей", 2, 15, 6)
        year_built = st.slider("Год постройки", 1870, 2024, 2000)
        year_remod_add = st.slider("Год ремонта", 1870, 2024, 2005)
        lot_area = st.number_input("Площадь участка (кв.футы)", 1000, 50000, 10000, step=100)
        overall_cond = st.slider("Общее состояние (1-10)", 1, 10, 5)
    
    # Дополнительные параметры
    with st.expander("Дополнительные параметры"):
        col3, col4 = st.columns(2)
        with col3:
            bedroom_abv_gr = st.slider("Спален над землей", 0, 8, 3)
            fireplaces = st.slider("Камины", 0, 4, 1)
        with col4:
            garage_area = st.number_input("Площадь гаража", 0, 1500, 500, step=50)
            mo_sold = st.slider("Месяц продажи", 1, 12, 6)
            yr_sold = st.slider("Год продажи", 2000, 2024, 2023)
    
    # Создаем DataFrame
    manual_data = pd.DataFrame([{
        'OverallQual': overall_qual,
        'GrLivArea': gr_liv_area,
        'TotalBsmtSF': total_bsmt_sf,
        'GarageCars': garage_cars,
        'FullBath': full_bath,
        'TotRmsAbvGrd': tot_rms_abv_grd,
        'YearBuilt': year_built,
        'YearRemodAdd': year_remod_add,
        'LotArea': lot_area,
        'OverallCond': overall_cond,
        'BedroomAbvGr': bedroom_abv_gr,
        'Fireplaces': fireplaces,
        'GarageArea': garage_area,
        'MoSold': mo_sold,
        'YrSold': yr_sold,
        '1stFlrSF': gr_liv_area * 0.6,  # Примерное значение
        '2ndFlrSF': gr_liv_area * 0.4   # Примерное значение
    }])
    
    return manual_data

# Основное содержимое
tab1, tab2, tab3 = st.tabs(["🏠 Предсказание", "📊 Анализ", "📈 Графики"])

# Инициализация session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'ids' not in st.session_state:
    st.session_state.ids = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None

# Вкладка 1: Предсказание
with tab1:
    st.markdown('<h2 class="sub-header">🔮 Предсказание цен на жилье</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Способ загрузки данных
        data_source = st.radio(
            "Выберите способ загрузки данных:",
            ["📤 Загрузить CSV файл", "📝 Ввести данные вручную", "🎲 Пример данных"]
        )
        
        data_to_predict = None
        data_display = None
        
        if data_source == "📤 Загрузить CSV файл":
            uploaded_file = st.file_uploader("Загрузите CSV файл с данными", type=['csv'], key="csv_uploader")
            
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    data_display = data.copy()
                    
                    st.success(f"✅ Файл успешно загружен: {uploaded_file.name}")
                    
                    with st.expander("👀 Просмотр данных", expanded=True):
                        st.dataframe(data.head(10))
                        st.write(f"**Размер данных:** {data.shape[0]} строк, {data.shape[1]} столбцов")
                    
                    data_to_predict = data
                
                except Exception as e:
                    st.error(f"Ошибка при чтении файла: {e}")
        
        elif data_source == "📝 Ввести данные вручную":
            manual_data = create_manual_input()
            
            if st.button("💾 Сохранить данные", key="save_manual"):
                st.session_state.manual_data = manual_data
                st.success("✅ Данные сохранены!")
                data_to_predict = manual_data
                data_display = manual_data.copy()
        
        else:  # Пример данных
            st.markdown("### Пример данных для тестирования")
            
            # Создаем пример данных
            example_data = pd.DataFrame([{
                'OverallQual': 7,
                'GrLivArea': 1500,
                'TotalBsmtSF': 1000,
                'GarageCars': 2,
                'FullBath': 2,
                'TotRmsAbvGrd': 6,
                'YearBuilt': 2000,
                'YearRemodAdd': 2005,
                'LotArea': 10000,
                'OverallCond': 5,
                'BedroomAbvGr': 3,
                'Fireplaces': 1,
                'GarageArea': 500,
                'MoSold': 6,
                'YrSold': 2023,
                '1stFlrSF': 900,
                '2ndFlrSF': 600
            }])
            
            st.dataframe(example_data)
            
            if st.button("🎯 Использовать пример", key="use_example"):
                data_to_predict = example_data
                data_display = example_data.copy()
                st.success("✅ Пример данных загружен!")
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Запуск предсказания")
        
        # Загружаем модель
        model = load_model(selected_model)
        
        if model is not None:
            st.success(f"✅ Модель загружена")
            
            if st.button("🚀 Запустить предсказание", type="primary", use_container_width=True, key="run_prediction"):
                if data_to_predict is not None:
                    with st.spinner("🤖 Выполняется предсказание..."):
                        try:
                            # Сохраняем ID если есть
                            if 'Id' in data_to_predict.columns:
                                ids = data_to_predict['Id']
                                X = data_to_predict.drop('Id', axis=1)
                            else:
                                ids = pd.Series(range(1, len(data_to_predict) + 1))
                                X = data_to_predict
                            
                            # Подготавливаем данные (только основные признаки)
                            X_prepared = prepare_basic_features(X)
                            
                            # Делаем предсказания
                            predictions = model.predict(X_prepared)
                            predictions = np.clip(predictions, 0, None)
                            
                            # Проверяем масштаб (если модель обучалась на log(y))
                            if predictions.mean() < 1000:
                                predictions = np.expm1(predictions)
                                predictions = np.clip(predictions, 0, None)
                            
                            # Сохраняем результаты
                            st.session_state.predictions = predictions
                            st.session_state.ids = ids
                            st.session_state.input_data = X_prepared
                            
                            st.success(f"✅ Предсказание завершено!")
                            
                        except Exception as e:
                            st.error(f"❌ Ошибка при предсказании: {e}")
                else:
                    st.warning("⚠️ Пожалуйста, сначала загрузите или создайте данные")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Показываем результаты если они есть
        if st.session_state.predictions is not None:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Результаты")
            
            predictions = st.session_state.predictions
            avg_price = predictions.mean()
            
            st.metric("Средняя цена", f"${avg_price:,.0f}")
            st.metric("Мин/Макс", f"${predictions.min():,.0f} / ${predictions.max():,.0f}")
            
            # Кнопка для скачивания
            results_df = pd.DataFrame({
                'Id': st.session_state.ids,
                'Predicted_Price': predictions
            })
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать результаты",
                data=csv,
                file_name="house_price_predictions.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_results"
            )
            st.markdown('</div>', unsafe_allow_html=True)

# Вкладка 2: Анализ
with tab2:
    st.markdown('<h2 class="sub-header">📊 Анализ предсказаний</h2>', unsafe_allow_html=True)
    
    if st.session_state.predictions is not None:
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
    
    if st.session_state.predictions is not None:
        # Выбор типа графика
        chart_type = st.selectbox(
            "Выберите тип графика:",
            ["Гистограмма распределения", "Box plot", "Сравнение признаков"],
            key="chart_type"
        )
        
        if chart_type == "Гистограмма распределения":
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
        
        elif chart_type == "Сравнение признаков" and st.session_state.input_data is not None:
            # Выбор признака для анализа
            if not st.session_state.input_data.empty:
                available_features = st.session_state.input_data.columns.tolist()
                
                if available_features:
                    selected_feature = st.selectbox("Выберите признак для анализа:", available_features)
                    
                    if selected_feature in st.session_state.input_data.columns:
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=st.session_state.input_data[selected_feature],
                            y=st.session_state.predictions,
                            mode='markers',
                            marker=dict(
                                size=10,
                                color=st.session_state.predictions,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title='Цена')
                            ),
                            text=[f"Цена: ${p:,.0f}<br>{selected_feature}: {x}" 
                                  for p, x in zip(st.session_state.predictions, 
                                               st.session_state.input_data[selected_feature])],
                            hoverinfo='text'
                        ))
                        
                        fig.update_layout(
                            title=f"Зависимость цены от {selected_feature}",
                            xaxis_title=selected_feature,
                            yaxis_title="Предсказанная цена ($)",
                            height=500,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("ℹ️ Сначала выполните предсказание на вкладке 'Предсказание'")

# Футер
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; margin-top: 2rem;'>
    © 2024 House Price Predictor. Использует основные признаки для быстрого предсказания.
    </div>
    """,
    unsafe_allow_html=True
)