import pandas as pd
import numpy as np
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- 💡 1. กำหนด Feature Engineering Functions ---

# Function ใหม่สำหรับ Capacity
def apply_capacity_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    epsilon = 1e-6
    if 'Enthalpy Dif.' not in df_out.columns:
        if 'Entering Air Enthalpy' in df_out.columns and 'Leaving Air Enthalpy' in df_out.columns:
            df_out['Enthalpy Dif.'] = df_out['Entering Air Enthalpy'] - df_out['Leaving Air Enthalpy']
        else: df_out['Enthalpy Dif.'] = 0
    df_out['Nozzle Dif.'] = df_out['Indoor Inlet DBT'] - df_out['Indoor Nozzle Temp.']
    df_out['Airflow Log'] = np.log1p(df_out['Airflow'])
    df_out['Dehum Log'] = np.log1p(df_out['Dehumidify Capacity'])
    df_out['I^2'] = df_out['TestUnit Current']**2
    df_out['Airflow Squared'] = df_out['Airflow'].pow(2)
    df_out['Enthalpy per current'] = df_out['Enthalpy Dif.'] / (df_out['TestUnit Current'] + epsilon)
    df_out['Discharge x Airflow'] = df_out['Discharge'] * df_out['Airflow']
    df_out['I^3'] = df_out['TestUnit Current']**3
    return df_out

# Function เดิมสำหรับ Power
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
    # สร้าง feature ใหม่ๆ โดยใช้ .get() เพื่อความปลอดภัยเผื่อบางคอลัมน์ไม่มี
    df_out['Current Log'] = np.log1p(df_out.get('TestUnit Current', 0))
    df_out['Current per Airflow'] = df_out.get('TestUnit Current', 0) / (df_out.get('Airflow', 0) + epsilon)
    df_out['Ratio Log (Current/Airflow)'] = np.log1p(df_out['Current per Airflow'])
    df_out['Temp Dif Squared'] = pd.Series(temp_dif).pow(2)
    if 'I^2' not in df_out.columns:
        df_out['I^2'] = df_out.get('TestUnit Current', 0)**2
    if 'Enthalpy per current' not in df_out.columns:
        df_out['Enthalpy per current'] = df_out['Enthalpy Dif.'] / (df_out.get('TestUnit Current', 0) + epsilon)
    df_out['COP Approx'] = df_out['Enthalpy Dif.'] / (df_out.get('TestUnit Current', 0) + epsilon)
    return df_out

# --- 💡 2. กำหนด Feature Lists ---
BASE_CAPACITY_FEATURES = ['Indoor Nozzle Temp.', 'Dehumidify Capacity', 'Airflow', 'TestUnit Current', 'R32', 'Cond. Out', 'Discharge']
ENGINEERED_CAPACITY_FEATURES = ['Nozzle Dif.', 'Airflow Log', 'Dehum Log', 'I^2', 'Airflow Squared', 'Enthalpy per current', 'Discharge x Airflow', 'I^3']
FEATURES_CAPACITY = BASE_CAPACITY_FEATURES + ENGINEERED_CAPACITY_FEATURES

BASE_POWER_FEATURES = ['TestUnit Current', 'R32', 'Cond. Out', 'Discharge', 'Airflow']
ENGINEERED_POWER_FEATURES = ['Airflow Log', 'Current Log', 'Current per Airflow', 'Ratio Log (Current/Airflow)', 'I^2', 'Temp Dif Squared', 'Enthalpy per current', 'COP Approx']
FEATURES_POWER = BASE_POWER_FEATURES + ENGINEERED_POWER_FEATURES

# STEP 1: ดึงข้อมูลจาก Google Sheets
try:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1w3s0zOMI1Id_SVapUWl-Ui16lEsJyETsltSxCiDy05M/edit")
    worksheet = sheet.worksheet("DATA")
    data = pd.DataFrame(worksheet.get_all_records())
    print("✅ ดึงข้อมูลสำเร็จ")
    data.columns = data.columns.str.strip()
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาด: {e}")
    exit()

# STEP 2: ทำความสะอาดข้อมูลดิบ
raw_features_needed = [
    'TestUnit Current', 'R32', 'Cond. Out', 'Discharge', 'Airflow', 'Dehumidify Capacity',
    'Entering Air Enthalpy', 'Leaving Air Enthalpy', 'Indoor Inlet DBT', 'Indoor Outlet DBT',
    'O.D Air D.B.T', 'Air Outlet ID', 'Indoor Nozzle Temp.'
]
targets = ['Capacity', 'Power']
columns_to_process = list(set(raw_features_needed + targets))

for col in columns_to_process:
    data[col] = data[col].replace('', np.nan)
data.dropna(subset=columns_to_process, inplace=True)
for col in columns_to_process:
    data[col] = pd.to_numeric(data[col].astype(str).str.replace(",", ""), errors='coerce')
print(f"✅ ข้อมูลหลังทำความสะอาดเหลือ {data.shape[0]} แถว")

# --- STEP 3: ทำ Feature Engineering ---
data = apply_capacity_engineering(data)
data = apply_power_engineering(data)
print("✅ ทำ Feature Engineering สำเร็จ")

# STEP 4: เทรนโมเดล
param_grid = { 'n_estimators': [100, 200], 'max_depth': [10, None], 'min_samples_leaf': [1, 2] }
base_rf = RandomForestRegressor(random_state=42)

# ----- เทรนโมเดล Capacity -----
print(f"\n--- Training Capacity Model with {len(FEATURES_CAPACITY)} features ---")
X_cap = data[FEATURES_CAPACITY]
y_cap = data['Capacity']
X_cap_train, X_cap_test, y_cap_train, y_cap_test = train_test_split(X_cap, y_cap, test_size=0.2, random_state=42)
grid_search_capacity = GridSearchCV(estimator=base_rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search_capacity.fit(X_cap_train, y_cap_train)
best_model_capacity = grid_search_capacity.best_estimator_
joblib.dump(best_model_capacity, 'model_capacity_new_eng.pkl')
print(f"✅ บันทึกโมเดล Capacity สำเร็จ")

# ----- เทรนโมเดล Power -----
print(f"\n--- Training Power Model with {len(FEATURES_POWER)} features ---")
X_pow = data[FEATURES_POWER]
y_pow = data['Power']
X_pow_train, X_pow_test, y_pow_train, y_pow_test = train_test_split(X_pow, y_pow, test_size=0.2, random_state=42)
grid_search_power = GridSearchCV(estimator=base_rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search_power.fit(X_pow_train, y_pow_train)
best_model_power = grid_search_power.best_estimator_
joblib.dump(best_model_power, 'model_power_final_eng.pkl')
print(f"✅ บันทึกโมเดล Power สำเร็จ")

