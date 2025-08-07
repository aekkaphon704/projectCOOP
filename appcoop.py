import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn

# --- 1. การตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="AC Performance Prediction",
    page_icon="❄️",
    layout="wide"
)

# --- CSS สำหรับตกแต่ง ---
st.markdown("""
<style>
.results-card { background-color: #F0F8FF; border-radius: 15px; padding: 25px; border: 1px solid #B0C4DE; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1); }
.results-card h3 { color: #003366; text-align: center; margin-bottom: 20px; }
.results-card .metric-container { display: flex; justify-content: space-around; margin-top: 15px; }
.results-card .metric { text-align: center; }
.results-card .metric-value { font-size: 2.5rem; font-weight: bold; color: #003366; }
.results-card .metric-label { font-size: 1rem; color: #5A5A5A; }
</style>
""", unsafe_allow_html=True)

# --- 2. 💡 ส่วน Logic (ต้องเหมือนกับ train_model.py ทุกประการ) ---
# Feature Lists
BASE_CAPACITY_FEATURES = ['Indoor Nozzle Temp.', 'Dehumidify Capacity', 'Airflow', 'TestUnit Current', 'R32', 'Cond. Out', 'Discharge']
ENGINEERED_CAPACITY_FEATURES = ['Nozzle Dif.', 'Airflow Log', 'Dehum Log', 'I^2', 'Airflow Squared', 'Enthalpy per current', 'Discharge x Airflow', 'I^3']
FEATURES_CAPACITY = BASE_CAPACITY_FEATURES + ENGINEERED_CAPACITY_FEATURES

BASE_POWER_FEATURES = ['TestUnit Current', 'R32', 'Cond. Out', 'Discharge', 'Airflow']
ENGINEERED_POWER_FEATURES = ['Airflow Log', 'Current Log', 'Current per Airflow', 'Ratio Log (Current/Airflow)', 'I^2', 'Temp Dif Squared', 'Enthalpy per current', 'COP Approx']
FEATURES_POWER = BASE_POWER_FEATURES + ENGINEERED_POWER_FEATURES

# Feature Engineering Functions
def apply_capacity_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    epsilon = 1e-6
    if 'Enthalpy Dif.' not in df_out.columns:
        if 'Entering Air Enthalpy' in df_out.columns and 'Leaving Air Enthalpy' in df_out.columns:
            df_out['Enthalpy Dif.'] = df_out['Entering Air Enthalpy'] - df_out['Leaving Air Enthalpy']
        else: df_out['Enthalpy Dif.'] = 0
    df_out['Nozzle Dif.'] = df_out.get('Indoor Inlet DBT', 0) - df_out.get('Indoor Nozzle Temp.', 0)
    df_out['Airflow Log'] = np.log1p(df_out.get('Airflow', 0))
    df_out['Dehum Log'] = np.log1p(df_out.get('Dehumidify Capacity', 0))
    df_out['I^2'] = df_out.get('TestUnit Current', 0)**2
    df_out['Airflow Squared'] = df_out.get('Airflow', 0)**2
    df_out['Enthalpy per current'] = df_out['Enthalpy Dif.'] / (df_out.get('TestUnit Current', 0) + epsilon)
    df_out['Discharge x Airflow'] = df_out.get('Discharge', 0) * df_out.get('Airflow', 0)
    df_out['I^3'] = df_out.get('TestUnit Current', 0)**3
    return df_out

def apply_power_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    epsilon = 1e-6
    if 'Enthalpy Dif.' not in df_out.columns:
        if 'Entering Air Enthalpy' in df_out.columns and 'Leaving Air Enthalpy' in df_out.columns:
            df_out['Enthalpy Dif.'] = df_out['Entering Air Enthalpy'] - df_out['Leaving Air Enthalpy']
        else: df_out['Enthalpy Dif.'] = 0
    if 'Indoor Inlet DBT' in df_out.columns and 'Indoor Outlet DBT' in df_out.columns:
        temp_dif = df_out['Indoor Inlet DBT'] - df_out['Indoor Outlet DBT']
    else: temp_dif = 0
    df_out['Current Log'] = np.log1p(df_out.get('TestUnit Current', 0))
    df_out['Current per Airflow'] = df_out.get('TestUnit Current', 0) / (df_out.get('Airflow', 0) + epsilon)
    df_out['Ratio Log (Current/Airflow)'] = np.log1p(df_out['Current per Airflow'])
    df_out['Temp Dif Squared'] = pd.Series(temp_dif).pow(2)
    if 'I^2' not in df_out.columns:
        df_out['I^2'] = df_out.get('TestUnit Current', 0)**2
    if 'Enthalpy per current' not in df_out.columns:
        df_out['Enthalpy per current'] = df_out['Enthalpy Dif.'] / (df_out.get('TestUnit Current', 0) + epsilon)
    if 'Airflow Log' not in df_out.columns:
         df_out['Airflow Log'] = np.log1p(df_out.get('Airflow', 0))
    df_out['COP Approx'] = df_out['Enthalpy Dif.'] / (df_out.get('TestUnit Current', 0) + epsilon)
    return df_out

# --- 3. ฟังก์ชันสำหรับโหลดโมเดล ---
@st.cache_resource
def load_models():
    models = {}
    try:
        models['capacity'] = joblib.load('model_capacity_new_eng.pkl')
        models['power'] = joblib.load('model_power_final_eng.pkl')
    except FileNotFoundError as e:
        st.error(f"ไม่พบไฟล์โมเดล: {e.filename}")
    return models

models = load_models()

# --- 4. ส่วนหัวและโลโก้ ---
col_title, col_empty, col_cmu, col_haier = st.columns([5, 1, 1, 1])
with col_title:
    st.title("❄️ AC Performance Prediction App")
with col_cmu:
    st.image("Logo CMU.jpg", width=80)
with col_haier:
    st.image("Logo Haier.png", width=80)
st.markdown("---")

# --- 5. จัด Layout และ Input ---
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.header("1. กรอกค่าพารามิเตอร์")
    
    RAW_FEATURES_NEEDED = [
        'R32', 'Cond. Out', 'Discharge', 'TestUnit Current', 'Airflow',
        'Dehumidify Capacity', 'Indoor Nozzle Temp.', 'Indoor Inlet DBT',
        'Indoor Outlet DBT', 'Entering Air Enthalpy', 'Leaving Air Enthalpy'
    ]
    
    input_col1, input_col2 = st.columns(2)
    input_data = {}
    
    half_way = (len(RAW_FEATURES_NEEDED) + 1) // 2
    
    with input_col1:
        for feature in RAW_FEATURES_NEEDED[:half_way]:
            input_data[feature] = st.number_input(f'{feature}', value=0.0, format="%.3f", key=feature)
            
    with input_col2:
        for feature in RAW_FEATURES_NEEDED[half_way:]:
            input_data[feature] = st.number_input(f'{feature}', value=0.0, format="%.3f", key=feature)

# --- 6. ส่วนทำนายผล ---
with col_output:
    st.header("2. ผลการทำนาย")
    if st.button('คำนวณและทำนายผล', use_container_width=True, type="primary"):
        if 'capacity' in models and 'power' in models:
            try:
                input_df_raw = pd.DataFrame([input_data])
                
                # --- เรียกใช้ Feature Engineering ทั้ง 2 ฟังก์ชัน ---
                engineered_df1 = apply_capacity_engineering(input_df_raw)
                input_df_engineered = apply_power_engineering(engineered_df1)
                
                # เลือก Feature ให้ตรงกับแต่ละโมเดล
                input_df_cap = input_df_engineered[FEATURES_CAPACITY]
                input_df_pow = input_df_engineered[FEATURES_POWER]
                
                # ทำนายผล
                capacity_prediction = models['capacity'].predict(input_df_cap)[0]
                power_prediction = models['power'].predict(input_df_pow)[0]
                cop_value = (capacity_prediction / power_prediction) if power_prediction > 0 else 0
                
                # แสดงผลลัพธ์ในการ์ด
                st.markdown(
                    f"""
                    <div class="results-card">
                        <h3>📈 ผลการทำนาย</h3>
                        <div class="metric-container">
                            <div class="metric"><div class="metric-label">Predicted Capacity</div><div class="metric-value">{capacity_prediction:.2f} W</div></div>
                            <div class="metric"><div class="metric-label">Predicted Power</div><div class="metric-value">{power_prediction:.2f} W</div></div>
                            <div class="metric"><div class="metric-label">Calculated COP</div><div class="metric-value">{cop_value:.3f}</div></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
        else:
            st.warning("ไม่สามารถโหลดโมเดลได้ครบถ้วน")
    else:

        st.info("กรุณากรอกข้อมูลด้านซ้ายและกดปุ่มเพื่อดูผลลัพธ์")
    st.markdown("---") # เพิ่มเส้นคั่นเพื่อความสวยงาม
st.write("🪄CMU INTERNSHIP x AC R&D HAIER JUB JUB")


