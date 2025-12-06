import streamlit as st
import pandas as pd
import numpy as np
import joblib

# sklearn
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Predictor")
st.write("Предсказание цен на дома с использованием обученной модели GradientBoostingRegressor")

# -----------------------------
# Загрузка обученного pipeline
# -----------------------------
@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load("house_price_pipeline.pkl")  # твой full_pipeline
        st.success("✅ Pipeline загружен")
        return pipeline
    except:
        st.error("❌ Pipeline не найден!")
        return None

full_pipeline = load_pipeline()

# -----------------------------
# Загрузка CSV
# -----------------------------
st.header("Загрузка данных")
uploaded_file = st.file_uploader("Загрузите CSV с данными для предсказания", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Данные успешно загружены:")
    st.dataframe(data.head())

    # Предсказания
    if full_pipeline is not None:
        try:
            X_pred = data.copy()
            if "SalePrice" in X_pred.columns:
                y_true = X_pred["SalePrice"]
                X_pred = X_pred.drop("SalePrice", axis=1)
            else:
                y_true = None

            y_pred = full_pipeline.predict(X_pred)

            st.subheader("🏡 Предсказанные цены")
            st.write(y_pred)

            # Метрики если есть SalePrice
            if y_true is not None:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

                st.subheader("📊 Метрики")
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"R²: {r2:.3f}")
                st.write(f"RMSLE: {rmsle:.4f}")

        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")

else:
    st.info("Загрузите CSV, чтобы получить предсказания")

# -----------------------------
# Ручной ввод для одного дома
# -----------------------------
st.header("📝 Ручной ввод данных для одного дома")

# Ввод основных числовых колонок
default_values = {}
default_values['OverallQual'] = st.slider("Общее качество (OverallQual)", 1, 10, 7)
default_values['GrLivArea'] = st.number_input("Жилая площадь (GrLivArea)", 500, 5000, 1500)
default_values['TotalBsmtSF'] = st.number_input("Площадь подвала (TotalBsmtSF)", 0, 3000, 1000)
default_values['YearBuilt'] = st.number_input("Год постройки (YearBuilt)", 1900, 2024, 2000)
default_values['LotArea'] = st.number_input("Площадь участка (LotArea)", 1000, 50000, 10000)
default_values['BedroomAbvGr'] = st.slider("Спален (BedroomAbvGr)", 0, 8, 3)

# Ввод категориальных колонок (попробуем взять минимальный набор)
default_values['MSZoning'] = st.selectbox("Зонирование (MSZoning)", ['RL', 'RM', 'C (all)', 'FV', 'RH'])
default_values['Street'] = st.selectbox("Тип улицы (Street)", ['Pave', 'Grvl'])
default_values['CentralAir'] = st.selectbox("Кондиционер (CentralAir)", ['Y', 'N'])
default_values['KitchenQual'] = st.selectbox("Качество кухни (KitchenQual)", ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
default_values['Neighborhood'] = st.selectbox("Район (Neighborhood)", ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt'])
default_values['BsmtQual'] = st.selectbox("Качество подвала (BsmtQual)", ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'])
default_values['GarageType'] = st.selectbox("Тип гаража (GarageType)", ['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'Basment', 'NA'])

# Кнопка предсказания
if st.button("🎯 Предсказать цену для этого дома"):
    if full_pipeline is not None:
        try:
            # Создаем DataFrame с одной строкой
            df_input = pd.DataFrame([default_values])
            y_pred_manual = full_pipeline.predict(df_input)[0]
            st.success(f"🏡 Предсказанная цена: ${y_pred_manual:,.0f}")
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
    else:
        st.error("Pipeline не загружен")
