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
    .main-header { font-size: 2.8rem; color: #1E3A8A; text-align: center; margin-bottom: 1rem; font-weight: 700; background: linear-gradient(45deg, #1E3A8A, #3B82F6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .sub-header { font-size: 1.8rem; color: #2563EB; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 2px solid #3B82F6; padding-bottom: 0.5rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; margin: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    .prediction-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 2rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1); }
    .info-card { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏠 AI House Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("### *Точное предсказание цен на недвижимость с помощью Machine Learning*")

# ========== Загрузка модели и препроцессора ==========
@st.cache_resource
def load_model():
    try:
        model = joblib.load('GB_model.pkl')
        st.success("✅ Модель загружена")
        return model
    except:
        st.error("❌ Модель GB_model.pkl не найдена")
        return None

@st.cache_resource
def load_preprocessor():
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        st.success("✅ Препроцессор загружен")
        return preprocessor
    except:
        st.error("❌ Препроцессор не найден")
        return None

model = load_model()
preprocessor = load_preprocessor()

# ========== Колонки модели ==========
drop_columns = ['Id', '1stFlrSF', '2ndFlrSF', 'ExterQual', 'BsmtFinSF1', 'GarageYrBlt', 
                'TotRmsAbvGrd', 'GarageCars', 'PoolQC', 'MasVnrArea', 'YearRemodAdd', 
                'FullBath', '3SsnPorch', 'LotShape', 'FireplaceQu', 'HalfBath', 
                'MasVnrType', 'BsmtFinType2', 'PavedDrive', 'BsmtCond', 'Foundation', 
                'KitchenAbvGr', 'RoofStyle', 'HouseStyle', 'GarageQual', 'RoofMatl', 
                'Electrical', 'BldgType']

numerical_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'LotArea',
                      'BedroomAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF']
categorical_features = ['MSZoning', 'Street', 'CentralAir', 'KitchenQual', 'Neighborhood',
                        'BsmtQual', 'GarageType']

# ========== Форма для ручного ввода ==========
st.header("📝 Ручной ввод данных")
default_values = {}

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

with st.expander("📋 Категориальные признаки"):
    default_values['MSZoning'] = st.selectbox("Зонирование (MSZoning)", ['RL', 'RM', 'C (all)', 'FV', 'RH'])
    default_values['Street'] = st.selectbox("Тип улицы (Street)", ['Pave', 'Grvl'])
    default_values['CentralAir'] = st.selectbox("Кондиционер (CentralAir)", ['Y', 'N'])
    default_values['KitchenQual'] = st.selectbox("Качество кухни (KitchenQual)", ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
    default_values['Neighborhood'] = st.selectbox("Район (Neighborhood)", ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt'])
    default_values['BsmtQual'] = st.selectbox("Качество подвала (BsmtQual)", ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
    default_values['GarageType'] = st.selectbox("Тип гаража (GarageType)", ['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'Basment', 'NA'])

# ========== Предсказание ==========
if st.button("🎯 Предсказать цену"):
    if model is None or preprocessor is None:
        st.error("Модель или препроцессор не загружены!")
        st.stop()
    
    df_input = pd.DataFrame([default_values])
    try:
        # Обработка через препроцессор
        X_processed = preprocessor.transform(df_input)
        prediction = model.predict(X_processed)[0]
        st.success(f"## 🏡 Предсказанная цена: **${prediction:,.0f}**")
    except Exception as e:
        st.error(f"Ошибка при предсказании: {str(e)}")
        