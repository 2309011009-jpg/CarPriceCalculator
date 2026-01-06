import streamlit as st
import pandas as pd
import numpy as np
import re
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def simplify_color(color_string):
    if pd.isna(color_string):
        return 'Unknown'
    color_string = str(color_string).lower()
    if 'black' in color_string: return 'Black'
    elif 'white' in color_string: return 'White'
    elif 'silver' in color_string: return 'Silver'
    elif 'gray' in color_string or 'grey' in color_string: return 'Gray'
    elif 'blue' in color_string: return 'Blue'
    elif 'red' in color_string: return 'Red'
    elif 'green' in color_string: return 'Green'
    elif 'brown' in color_string or 'bronze' in color_string or 'gold' in color_string: return 'Brown/Gold'
    elif 'yellow' in color_string or 'orange' in color_string: return 'Yellow/Orange'
    elif 'purple' in color_string: return 'Purple'
    elif 'beige' in color_string or 'tan' in color_string: return 'Beige/Tan'
    else: return 'Other'

def simplify_transmission(trans_string):
    if pd.isna(trans_string):
        return 'Automatic'
    trans_string = str(trans_string).lower()
    if 'manual' in trans_string or 'm/t' in trans_string:
        return 'Manual'
    elif 'cvt' in trans_string:
        return 'CVT'
    # Covers A/T, Automatic, Dual Shift, Auto-Shift, etc.
    return 'Automatic'

# -----------------------------------------------------------------------------
# 2. DATA LOADING & TRAINING (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv("used_cars.csv")
    except FileNotFoundError:
        st.error("File 'used_cars.csv' not found. Ensure it is in the same directory.")
        return None, None

    # --- Preprocessing ---
    df['car_age'] = 2024 - df['model_year']
    
    # Mileage
    df['milage'] = df['milage'].astype(str).str.replace(' mi.', '', regex=False).str.replace(',', '', regex=False)
    df['milage'] = pd.to_numeric(df['milage'], errors='coerce')

    # Fuel Type
    df['fuel_type'] = df['fuel_type'].replace(['not supported', '–'], 'Electric')
    df['fuel_type'] = df['fuel_type'].replace('Plug-In Hybrid', 'Hybrid')
    df['fuel_type'] = df['fuel_type'].replace('E85 Flex Fuel', 'Gasoline')
    df['fuel_type'] = df['fuel_type'].fillna('Electric')

    # Engine Extraction
    df['horsepower'] = df['engine'].apply(lambda x: float(re.search(r'(\d+\.?\d*)HP', str(x)).group(1)) if re.search(r'(\d+\.?\d*)HP', str(x)) else None)
    df['engine_displacement_L'] = df['engine'].apply(lambda x: float(re.search(r'(\d+\.?\d*)L', str(x)).group(1)) if re.search(r'(\d+\.?\d*)L', str(x)) else None)
    df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)
    df['engine_displacement_L'].fillna(df['engine_displacement_L'].mean(), inplace=True)

    # Simplified Categoricals
    df['transmission'] = df['transmission'].apply(simplify_transmission)
    df['ext_col'] = df['ext_col'].apply(simplify_color)
    df['int_col'] = df['int_col'].apply(simplify_color)
    df['accident'].fillna('None reported', inplace=True)
    df['clean_title'].fillna('Not Reported', inplace=True)
    
    # Price
    df['price'] = df['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])

    # --- Feature Selection ---
    features = ['brand', 'car_age', 'milage', 'fuel_type', 'transmission', 'horsepower', 
                'engine_displacement_L', 'ext_col', 'int_col', 'accident', 'clean_title']
    
    X = df[features].copy()
    y = df['price']

    # --- Outlier Handling ---
    num_cols = ['milage', 'horsepower', 'engine_displacement_L', 'car_age']
    for col in num_cols:
        Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        IQR = Q3 - Q1
        X[col] = X[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    # --- CatBoost Training ---
    cat_features = ['brand', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.08, verbose=0, random_state=42)
    model.fit(X_train, y_train, cat_features=cat_features)

    unique_brands = sorted(df['brand'].unique().tolist())
    
    return model, unique_brands

# -----------------------------------------------------------------------------
# 3. STREAMLIT UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚙")
    st.title("🚙 Used Car Price Predictor")
    st.markdown("Enter details below to estimate the current market value.")

    with st.spinner("Preparing model..."):
        model, brands = train_model()

    if model is None: return

    # --- SIDEBAR INPUTS ---
    st.sidebar.header("Vehicle Specifications")
    brand = st.sidebar.selectbox("Brand", brands)
    car_age = st.sidebar.slider("Car Age (Years)", 0, 30, 5)
    mileage = st.sidebar.number_input("Mileage (mi)", value=40000, step=1000)
    
    # Transmission simplified in UI
    trans = st.sidebar.selectbox("Transmission", ["Automatic", "Manual", "CVT"])
    
    fuel = st.sidebar.selectbox("Fuel Type", ["Gasoline", "Hybrid", "Electric", "Diesel"])
    hp = st.sidebar.number_input("Horsepower (HP)", value=200.0)
    disp = st.sidebar.number_input("Engine Displacement (L)", value=2.0, step=0.1)
    
    st.sidebar.subheader("Appearance & History")
    colors = ['Black', 'White', 'Silver', 'Gray', 'Blue', 'Red', 'Green', 'Brown/Gold', 'Yellow/Orange', 'Purple', 'Beige/Tan', 'Other']
    ext_col = st.sidebar.selectbox("Exterior Color", colors)
    int_col = st.sidebar.selectbox("Interior Color", colors)
    
    acc_input = st.sidebar.radio("Accident Reported?", ["None reported", "At least 1 accident/damage recorded"])
    # Map back to exact training strings
    acc_val = "None reported" if "None" in acc_input else "At least 1 accident or damage reported"
    
    title_input = st.sidebar.radio("Clean Title?", ["Yes", "Not Reported"])

    # --- PREDICTION ---
    if st.button("Predict Price", type="primary"):
        # Create input dataframe
        input_df = pd.DataFrame({
            'brand': [brand], 'car_age': [car_age], 'milage': [mileage], 
            'fuel_type': [fuel], 'transmission': [trans], 'horsepower': [hp],
            'engine_displacement_L': [disp], 'ext_col': [ext_col], 
            'int_col': [int_col], 'accident': [acc_val], 'clean_title': [title_input]
        })
        
        prediction = model.predict(input_df)[0]
        
        st.divider()
        st.metric(label="Estimated Price", value=f"${max(0, prediction):,.2f}")
        
        # Breakdown expander
        with st.expander("Show detailed features"):
            st.write(input_df)

if __name__ == '__main__':
    main()