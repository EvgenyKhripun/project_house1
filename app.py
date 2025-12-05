import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Predictor")
st.write("Предсказание цен на дома с использованием обученной модели GradientBoostingRegressor")

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700
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

@st.cache_resource
def load_model():
    try:
        model = joblib.load('GB_model.pkl')
        st.success("✅ Модель GradientBoostingRegressor загружена")
        return model
    except:
        st.error("❌ Модель GB_model.pkl не найдена")
        return None

model = load_model()

def load_preprocessor():# <==========================
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        st.success("✅ Препроцессор загружен")
        return preprocessor
    except:
        st.error("❌ Препроцессор не найден")
        return None

preprocessor = load_preprocessor()
# ========== ОПРЕДЕЛЕНИЕ КОЛОНОК ==========
# Колонки, которые были удалены при обучении
drop_columns = [
    'Id', '1stFlrSF', '2ndFlrSF', 'ExterQual', 'BsmtFinSF1', 'GarageYrBlt', 
    'TotRmsAbvGrd', 'GarageCars', 'PoolQC', 'MasVnrArea', 'YearRemodAdd', 
    'FullBath', '3SsnPorch', 'LotShape', 'FireplaceQu', 'HalfBath', 
    'MasVnrType', 'BsmtFinType2', 'PavedDrive', 'BsmtCond', 'Foundation', 
    'KitchenAbvGr', 'RoofStyle', 'HouseStyle', 'GarageQual', 'RoofMatl', 
    'Electrical', 'BldgType'
]

# Все колонки из оригинального датасета
all_original_columns = [
    'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
    'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
    'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
    'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
    'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
    'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
    'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
    'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
    'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
    'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
    'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
    'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
    'SaleCondition'
]

# Колонки, которые остаются после удаления
remaining_columns = [col for col in all_original_columns if col not in drop_columns]

# Разделение на числовые и категориальные (оригинальные типы)
numerical_columns_original = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
    'YearBuilt', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
]

categorical_columns_original = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
    'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
    'SaleType', 'SaleCondition'
]

# Удаляем те, что в drop_columns
numerical_features = [col for col in numerical_columns_original if col not in drop_columns]
categorical_features = [col for col in categorical_columns_original if col not in drop_columns]

st.info(f"Модель использует {len(numerical_features)} числовых и {len(categorical_features)} категориальных признаков")

# ========== СОЗДАНИЕ ПРЕПРОЦЕССОРА ==========
def create_preprocessor():
    """Создает препроцессор такой же, как при обучении"""
    
    # Имьютер для заполнения пропусков
    imputer = ColumnTransformer(
        transformers=[
            ("numerical_features", SimpleImputer(strategy="median"), numerical_features),
            ("categorical_features", SimpleImputer(strategy="most_frequent"), categorical_features)
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )
    
    # Трансформер для удаления колонок
    imputer_drop = ColumnTransformer(
        transformers=[("drop", "drop", drop_columns)],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )
    
    # Кодировщик и скейлер
    encoder_and_scaler = ColumnTransformer(
        transformers=[
            ('encoder', ce.CatBoostEncoder(), categorical_features),
            ('scaler', StandardScaler(), numerical_features)
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )
    
    # Полный пайплайн
    preprocessor = Pipeline([
        ('imputer_drop', imputer_drop),
        ("imputer", imputer),
        ("encoder_and_scaler", encoder_and_scaler)
    ])
    
    return preprocessor

# ========== ФОРМА ДЛЯ РУЧНОГО ВВОДА ==========
st.header("📝 Ручной ввод данных")

# Создаем словарь для значений по умолчанию
default_values = {}

# Основные числовые признаки
col1, col2, col3 = st.columns(3)

with col1:
    default_values['OverallQual'] = st.slider("Общее качество (OverallQual)", 1, 10, 7)
    default_values['GrLivArea'] = st.number_input("Жилая площадь (GrLivArea)", 500, 5000, 1500)
    default_values['TotalBsmtSF'] = st.number_input("Площадь подвала (TotalBsmtSF)", 0, 3000, 1000)
    
with col2:
    default_values['YearBuilt'] = st.number_input("Год постройки (YearBuilt)", 1900, 2024, 2000)
    default_values['LotArea'] = st.number_input("Площадь участка (LotArea)", 1000, 50000, 10000)
    default_values['BedroomAbvGr'] = st.slider("Спален (BedroomAbvGr)", 0, 8, 3)
    
with col3:
    default_values['Fireplaces'] = st.slider("Камины (Fireplaces)", 0, 4, 1)
    default_values['GarageArea'] = st.number_input("Площадь гаража (GarageArea)", 0, 1500, 500)
    default_values['WoodDeckSF'] = st.number_input("Площадь террасы (WoodDeckSF)", 0, 1000, 0)

# Категориальные признаки
with st.expander("📋 Категориальные признаки"):
    cat_col1, cat_col2 = st.columns(2)
    
    with cat_col1:
        default_values['MSZoning'] = st.selectbox("Зонирование (MSZoning)", 
            ['RL', 'RM', 'C (all)', 'FV', 'RH'])
        default_values['Street'] = st.selectbox("Тип улицы (Street)", ['Pave', 'Grvl'])
        default_values['CentralAir'] = st.selectbox("Кондиционер (CentralAir)", ['Y', 'N'])
        default_values['KitchenQual'] = st.selectbox("Качество кухни (KitchenQual)", 
            ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
        
    with cat_col2:
        default_values['Neighborhood'] = st.selectbox("Район (Neighborhood)", 
            ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt'])
        default_values['BsmtQual'] = st.selectbox("Качество подвала (BsmtQual)", 
            ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
        default_values['GarageType'] = st.selectbox("Тип гаража (GarageType)", 
            ['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'Basment', 'NA'])

# Остальные значения по умолчанию
for col in numerical_features:
    if col not in default_values:
        if col == 'MSSubClass': default_values[col] = 60
        elif col == 'LotFrontage': default_values[col] = 70.0
        elif col == 'OverallCond': default_values[col] = 5
        elif col == 'MasVnrArea': default_values[col] = 0.0
        elif col == 'BsmtFinSF2': default_values[col] = 0.0
        elif col == 'BsmtUnfSF': default_values[col] = 500.0
        elif col == 'LowQualFinSF': default_values[col] = 0.0
        elif col == 'BsmtFullBath': default_values[col] = 0.0
        elif col == 'BsmtHalfBath': default_values[col] = 0.0
        elif col == 'HalfBath': default_values[col] = 1.0
        elif col == 'KitchenAbvGr': default_values[col] = 1.0
        elif col == 'ScreenPorch': default_values[col] = 0.0
        elif col == 'PoolArea': default_values[col] = 0.0
        elif col == 'MiscVal': default_values[col] = 0.0
        elif col == 'MoSold': default_values[col] = 6.0
        elif col == 'YrSold': default_values[col] = 2023.0
        else: default_values[col] = 0.0

for col in categorical_features:
    if col not in default_values:
        if col == 'Alley': default_values[col] = 'NA'
        elif col == 'LandContour': default_values[col] = 'Lvl'
        elif col == 'Utilities': default_values[col] = 'AllPub'
        elif col == 'LotConfig': default_values[col] = 'Inside'
        elif col == 'LandSlope': default_values[col] = 'Gtl'
        elif col == 'Condition1': default_values[col] = 'Norm'
        elif col == 'Condition2': default_values[col] = 'Norm'
        elif col == 'RoofStyle': default_values[col] = 'Gable'
        elif col == 'RoofMatl': default_values[col] = 'CompShg'
        elif col == 'Exterior1st': default_values[col] = 'VinylSd'
        elif col == 'Exterior2nd': default_values[col] = 'VinylSd'
        elif col == 'ExterCond': default_values[col] = 'TA'
        elif col == 'BsmtExposure': default_values[col] = 'No'
        elif col == 'BsmtFinType1': default_values[col] = 'Unf'
        elif col == 'Heating': default_values[col] = 'GasA'
        elif col == 'HeatingQC': default_values[col] = 'TA'
        elif col == 'Functional': default_values[col] = 'Typ'
        elif col == 'GarageFinish': default_values[col] = 'Unf'
        elif col == 'GarageCond': default_values[col] = 'TA'
        elif col == 'Fence': default_values[col] = 'NA'
        elif col == 'MiscFeature': default_values[col] = 'NA'
        elif col == 'SaleType': default_values[col] = 'WD'
        elif col == 'SaleCondition': default_values[col] = 'Normal'
        else: default_values[col] = 'NA'

# Кнопка предсказания
if st.button("🎯 Предсказать цену", type="primary", use_container_width=True):
    if model is None:
        st.error("Модель не загружена!")
        st.stop()
    
    if preprocessor is None:
        st.error("Препроцессор не загружен!")
        st.stop()
    
    with st.spinner("Обрабатываю данные..."):
        try:
            # Создаем DataFrame с оригинальными колонками
            input_data = {col: default_values.get(col, None) for col in all_original_columns}
            df_input = pd.DataFrame([input_data])
            df_input['Id'] = 999
            
            # Обработка через препроцессор
            X_processed = preprocessor.transform(df_input)
            
            # Предсказание
            prediction = model.predict(X_processed)[0]
            st.success(f"## 🏡 Предсказанная цена: **${prediction:,.0f}**")
            
        except Exception as e:
            # Если модель не смогла предсказать, используем упрощенную формулу
            st.info("Использую упрощенное предсказание на основе основных признаков")
            simple_pred = (default_values['OverallQual'] * 10000 +
                           default_values['GrLivArea'] * 50 +
                           default_values['YearBuilt'] * 100)
            st.success(f"## 🏡 Ориентировочная цена: **${simple_pred:,.0f}**")
            st.error(f"Ошибка: {str(e)[:200]}")


st.header("📤 Загрузите CSV файл с данными для предсказания")

uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])
if uploaded_file is not None:
    if model is None:
        st.error("Модель не загружена!")
        st.stop()

    try:
        df_csv = pd.read_csv(uploaded_file)
        st.write("✅ Файл загружен. Превью данных:")
        st.dataframe(df_csv.head())

        # Добавляем недостающие колонки
        for col in remaining_columns:
            if col not in df_csv.columns:
                df_csv[col] = np.nan

        # Заполняем пропуски числовых и категориальных признаков
        for col in numerical_features:
            if col in df_csv.columns:
                df_csv[col] = df_csv[col].fillna(df_csv[col].median() if df_csv[col].notna().any() else 0)

        for col in categorical_features:
            if col in df_csv.columns:
                df_csv[col] = df_csv[col].fillna('NA')

        # Удаляем ненужные колонки
        df_csv_processed = df_csv.drop(columns=[col for col in drop_columns if col in df_csv.columns])

        # Приводим порядок колонок к нужному
        feature_order = numerical_features + categorical_features
        X_model = df_csv_processed[feature_order]

        # Предсказание
        predictions = model.predict(X_model)
        df_csv['PredictedPrice'] = predictions

        st.success("✅ Предсказания выполнены")
        st.dataframe(df_csv[['PredictedPrice']].head())

        # Кнопка для скачивания результатов
        csv = df_csv.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Скачать результаты с предсказаниями",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Ошибка при предсказании: {str(e)}")
        