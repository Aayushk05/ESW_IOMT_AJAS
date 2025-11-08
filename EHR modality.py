# Step 1: Install and Import Libraries
print("--- Installing Libraries ---")
!pip install lightgbm -q
import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import zipfile
from datetime import datetime
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Step 2: Clean up any previous data to ensure a fresh start
print("\n--- Cleaning up previous data... ---")
!rm -rf /content/fhir/
!rm -rf /content/output/

# Step 3: Mount Google Drive
print("\n--- Mounting Google Drive... ---")
drive.mount('/content/drive')

# Step 4: Unpack the archive (fhir.zip)
zip_path = '/content/drive/My Drive/fhir.zip'
print(f"\n--- Unpacking file from: {zip_path} ---")
if os.path.exists(zip_path):
    # Use the 'unzip' command for .zip files
    # The -q flag makes it "quiet" to avoid listing every single file
    !unzip -q "{zip_path}" -d /content/
    print("--- fhir.zip unpacked successfully! ---")
else:
    print(f"--- WARNING: {zip_path} not found. Skipping. ---")

# Step 5: Consolidate and Verify the Final Directory
print("\n--- All archives processed! Verifying final directory... ---")
# This is a safety check. Sometimes archives create nested folders.
# This code finds the final 'fhir' folder and moves it to the correct location.
if os.path.exists('/content/output/fhir'):
    !mv /content/output/fhir /content/fhir
    !rm -rf /content/output

if os.path.exists('/content/fhir'):
    print("Final 'fhir' directory found. First 5 files:")
    !ls /content/fhir | head -n 5
else:
    print("--- ERROR: Final 'fhir' directory not found. Please check the contents of your archive. ---")

import os
non_empty = [f for f in os.listdir('/content/fhir/') if os.path.getsize(os.path.join('/content/fhir/', f)) > 0]
print(f"{len(non_empty)} non-empty files found.")

# This block parses all patient files and saves them to disk.
# It does NOT create the final DataFrame, which prevents the RAM crash.

# --- INSTALL LIBRARIES & IMPORT MODULES ---
!pip install pyarrow -q
import os
import json
import pandas as pd
import numpy as np
import shutil  # <-- IMPORTED SHUTIL
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- DEFINE PARSING FUNCTION (with medication/procedure history) ---
def parse_fhir_bundle_24h(file_path):
    try:
        if os.path.getsize(file_path) == 0: return None
        with open(file_path, 'r') as f: bundle = json.load(f)
        if not isinstance(bundle, dict) or 'entry' not in bundle: return None
 
        patient_id = os.path.basename(file_path).split('.')[0]
        patient_features = {'patient_id': patient_id}
        observations, conditions, medications, procedures = [], [], [], []

        for entry in bundle.get('entry', []):
            resource = entry.get('resource', {})
            r_type = resource.get('resourceType')

            if r_type == 'Patient':
                patient_features['gender'] = resource.get('gender')
                birth_date = resource.get('birthDate')
                if birth_date:
                    try:
                        patient_features['age'] = (datetime(2025, 1, 1) - datetime.strptime(birth_date, '%Y-%m-%d')).days // 365
                    except: patient_features['age'] = None
            elif r_type == 'Condition':
                conditions.append(resource.get('code', {}).get('text', '').lower())
            elif r_type == 'Observation':
                obs_time = resource.get('effectiveDateTime')
                if obs_time: observations.append((datetime.fromisoformat(obs_time.replace('Z', '+00:00')), resource))
            elif r_type == 'MedicationRequest':
                medications.append(resource.get('medicationCodeableConcept', {}).get('text', '').lower())
            elif r_type == 'Procedure':
                procedures.append(resource.get('code', {}).get('text', '').lower())

        if not observations: return None
        observations.sort(key=lambda x: x[0])

        obs_map = {'8480-6':'sbp','8462-4':'dbp','8867-4':'heart_rate','9279-1':'respiratory_rate','59408-5':'spo2','8310-5':'temperature','6690-2':'wbc_count','718-7':'hemoglobin','4544-3':'hematocrit','777-3':'platelet_count','2093-3':'cholesterol_total','2571-8':'triglycerides','18262-6':'cholesterol_ldl','2085-9':'cholesterol_hdl','72514-3':'pain_score','48643-1':'hba1c','55758-7':'phq2_score','70274-6':'gad7_score','8302-2':'height_m','29463-7':'weight_kg'}

        patient_rows = []
        for obs_time, obs in observations:
            row = patient_features.copy()
            row['obs_time'] = obs_time
            if 'component' in obs:
                for comp in obs['component']:
                    code = comp.get('code', {}).get('coding', [{}])[0].get('code')
                    if code in obs_map: row[obs_map[code]] = comp.get('valueQuantity', {}).get('value')
            else:
                code = obs.get('code', {}).get('coding', [{}])[0].get('code')
                if code in obs_map: row[obs_map[code]] = obs.get('valueQuantity', {}).get('value')

            row['history_hypertension'] = int(any('hypertension' in c for c in conditions))
            row['history_diabetes'] = int(any('diabetes' in c for c in conditions))
            row['history_chf'] = int(any('congestive heart failure' in c for c in conditions))
            row['history_copd'] = int(any('copd' in c for c in conditions))
            row['history_anemia'] = int(any('anemia' in c for c in conditions))
            row['history_hyperlipidemia'] = int(any('hyperlipidemia' in c for c in conditions))
            row['history_cad'] = int(any('coronary artery disease' in c for c in conditions))
            row['history_ckd'] = int(any('chronic kidney disease' in c for c in conditions))
            row['history_obesity'] = int(any('obesity' in c for c in conditions))
            row['history_asthma'] = int(any('asthma' in c for c in conditions))
            row['history_pneumonia'] = int(any('pneumonia' in c for c in conditions))
            row['history_flu'] = int(any('influenza' in c for c in conditions))
            row['on_statin'] = int(any('statin' in m for m in medications))
            row['on_metformin'] = int(any('metformin' in m for m in medications))
            row['on_aspirin'] = int(any('aspirin' in m for m in medications))
            row['on_lisinopril'] = int(any('lisinopril' in m for m in medications))
            row['history_cabg'] = int(any('cabg' in p or 'coronary artery bypass' in p for p in procedures))
            row['history_appendectomy'] = int(any('appendectomy' in p for p in procedures))
            patient_rows.append(row)
        return patient_rows
    except:
        return None

def process_and_save(file_path, output_dir):
    patient_data = parse_fhir_bundle_24h(file_path)
    if patient_data:
        patient_id = os.path.basename(file_path).split('.')[0]
        pd.DataFrame(patient_data).to_parquet(os.path.join(output_dir, f'{patient_id}.parquet'))

# --- SETUP DIRECTORIES AND RUN PARALLEL PROCESSING ---
data_directory = '/content/fhir/'
temp_output_dir = '/content/temp_parquet/'
# V-- THIS IS THE CORRECTED LINE --V
if os.path.exists(temp_output_dir):
    shutil.rmtree(temp_output_dir)
os.makedirs(temp_output_dir)

all_json_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.json')]
print(f"Total candidate JSON files: {len(all_json_files)}")

print("\n--- Parsing in parallel and saving to temporary files... ---")
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(process_and_save, fp, temp_output_dir): fp for fp in all_json_files}
    for i, future in enumerate(as_completed(futures), 1):
        if i % 1000 == 0: print(f"Processed {i} of {len(all_json_files)} files...")

print("\n✅ --- All files parsed and saved to disk. Ready for chunked processing. ---")

import pandas as pd
import numpy as np
import os
import shutil
import gc

# --- HELPER FUNCTION TO REDUCE MEMORY USAGE ---
def optimize_memory(df):
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if 'float' in str(df[col].dtype):
           df[col] = pd.to_numeric(df[col], downcast='float')
        elif 'int' in str(df[col].dtype):
           df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

# --- PROCESS FILES IN CHUNKS ---
print("--- Processing data with instantaneous features, trend features, and multi-timeframe labels... ---")
temp_output_dir = '/content/temp_parquet/'
all_parquet_files = [os.path.join(temp_output_dir, f) for f in os.listdir(temp_output_dir)]
chunk_size = 4000
processed_chunks = []

essential_columns = [ 'sbp', 'dbp', 'heart_rate', 'respiratory_rate', 'wbc_count', 'hemoglobin', 'hematocrit', 'platelet_count', 'cholesterol_total', 'triglycerides', 'cholesterol_ldl', 'cholesterol_hdl', 'hba1c', 'pain_score', 'phq2_score', 'gad7_score', 'height_m', 'weight_kg', 'bun', 'sodium', 'potassium', 'glucose', 'bilirubin', 'temperature', 'creatinine' ]

for i in range(0, len(all_parquet_files), chunk_size):
    chunk_files = all_parquet_files[i:i + chunk_size]
    if not chunk_files:
        continue

    df_chunk = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    print(f"Processing chunk {i//chunk_size + 1}...")

    df_processed = df_chunk.copy()
    for col in essential_columns:
        if col not in df_processed.columns:
            df_processed[col] = np.nan

    # --- TREND-BASED FEATURE ENGINEERING ---
    df_processed['obs_time'] = pd.to_datetime(df_processed['obs_time'])
    df_processed = df_processed.sort_values(['patient_id', 'obs_time']).set_index('obs_time')

    vitals_to_track = ['heart_rate', 'sbp', 'dbp', 'respiratory_rate', 'temperature', 'glucose', 'wbc_count', 'hemoglobin']
    for vital in vitals_to_track:
        if vital in df_processed.columns:
            # Changed '3H' to '3h' to fix the FutureWarning
            rolling_window = df_processed.groupby('patient_id')[vital].rolling(window='3h')
            df_processed[f'{vital}_avg_3hr'] = rolling_window.mean().reset_index(0, drop=True)
            df_processed[f'{vital}_std_3hr'] = rolling_window.std().reset_index(0, drop=True)

    df_processed.reset_index(inplace=True)

    # Forward-fill values within each patient group
    # Using a more modern approach to avoid the DeprecationWarning
    cols_to_fill = df_processed.columns.difference(['patient_id'])
    df_processed[cols_to_fill] = df_processed.groupby('patient_id')[cols_to_fill].ffill()

    # Fill any remaining NaNs (e.g., at the very start of a patient's record) with 0
    df_processed.fillna(0, inplace=True)

    # --- STANDARD & DELTA FEATURE ENGINEERING ---
    if 'gender' in df_processed.columns:
        df_processed = pd.get_dummies(df_processed, columns=['gender'], drop_first=True, dtype=int)

    safe_height_sq = (df_processed['height_m'].replace(0, np.nan))
    safe_sbp = df_processed['sbp'].replace(0, np.nan)
    safe_creatinine = df_processed['creatinine'].replace(0, np.nan)

    df_processed['map'] = ((2 * df_processed['dbp']) + df_processed['sbp']) / 3
    df_processed['shock_index'] = df_processed['heart_rate'] / safe_sbp

    for col in ['sbp', 'dbp', 'heart_rate', 'wbc_count', 'hemoglobin', 'map']:
        if col in df_processed.columns:
            # This line will now work correctly
            df_processed[f'delta_{col}'] = df_processed.groupby('patient_id')[col].diff().fillna(0)

    # --- MULTI-TIMEFRAME LABEL ENGINEERING ---
    # (The rest of your label creation logic remains the same)
    conditions = {
        'sepsis': [ ((df_processed['heart_rate'] > 110) & (df_processed['wbc_count'] > 14)), ((df_processed['heart_rate'] > 100) & (df_processed['wbc_count'] > 12)), ((df_processed['heart_rate'] > 90) & (df_processed['wbc_count'] > 11)) ],
        'anemia': [ (df_processed['hemoglobin'] < 10), (df_processed['hemoglobin'] < 12), (df_processed['hemoglobin'] < 13) ],
        'hyperlipidemia': [ ((df_processed['cholesterol_total'] > 240) | (df_processed['cholesterol_ldl'] > 160)), ((df_processed['cholesterol_total'] > 200) | (df_processed['cholesterol_ldl'] > 130)), ((df_processed['cholesterol_total'] > 190) | (df_processed['cholesterol_ldl'] > 120)) ],
        'mi': [ ((df_processed.get('history_cad', 0) == 1) & (df_processed['pain_score'] > 7)), ((df_processed.get('history_cad', 0) == 1) & (df_processed['pain_score'] > 5)), ((df_processed.get('history_cad', 0) == 1) & (df_processed['pain_score'] > 3)) ],
        'stroke': [ ((df_processed.get('history_hypertension', 0) == 1) & (df_processed['age'] > 70) & (df_processed['sbp'] > 160)), ((df_processed.get('history_hypertension', 0) == 1) & (df_processed['age'] > 60)), (df_processed['age'] > 55) ],
        'depression': [ (df_processed.get('phq2_score', 0) > 4), (df_processed.get('phq2_score', 0) > 2), (df_processed.get('phq2_score', 0) > 1) ],
        'pneumonia': [ ((df_processed['respiratory_rate'] > 24) & (df_processed['wbc_count'] > 13)), ((df_processed['respiratory_rate'] > 20) & (df_processed['wbc_count'] > 12)), ((df_processed['respiratory_rate'] > 18) & (df_processed['wbc_count'] > 11)) ],
       'chf_exacerbation': [ ((df_processed.get('history_chf', 0) == 1) & (df_processed['respiratory_rate'] > 22)), ((df_processed.get('history_chf', 0) == 1) & (df_processed['respiratory_rate'] > 20)), ((df_processed.get('history_chf', 0) == 1) & (df_processed['respiratory_rate'] > 18)) ],
        'hypertension': [ ((df_processed['sbp'] > 160) | (df_processed['dbp'] > 100)), ((df_processed['sbp'] > 140) | (df_processed['dbp'] > 90)), ((df_processed['sbp'] > 130) | (df_processed['dbp'] > 85)) ],
        'diabetes': [ (df_processed['glucose'] > 200), ((df_processed['hba1c'] > 6.5) | (df_processed['glucose'] > 126)), (df_processed['glucose'] > 110) ],
        'hypoglycemia': [ (df_processed['glucose'] < 60), (df_processed['glucose'] < 70), (df_processed['glucose'] < 80) ],
        'aki': [ (df_processed['creatinine'] > (df_processed['creatinine'].median() + 0.5)), (df_processed['creatinine'] > (df_processed['creatinine'].median() + 0.3)), (df_processed['creatinine'] > (df_processed['creatinine'].median() + 0.2)) ],
        'tachycardia': [ (df_processed['heart_rate'] > 120), (df_processed['heart_rate'] > 100), (df_processed['heart_rate'] > 95) ],
        'bradycardia': [ (df_processed['heart_rate'] < 50), (df_processed['heart_rate'] < 60), (df_processed['heart_rate'] < 65) ],
        'hypotension': [ (df_processed['sbp'] < 85), (df_processed['sbp'] < 90), (df_processed['sbp'] < 95) ],
        'acute_bronchitis': [ ((df_processed['respiratory_rate'] > 22) & (df_processed['wbc_count'] < 10)), ((df_processed['respiratory_rate'] > 20) & (df_processed['wbc_count'] < 12)), ((df_processed['respiratory_rate'] > 18) & (df_processed['wbc_count'] < 12)) ],
        'anxiety': [ (df_processed.get('gad7_score', 0) > 15), (df_processed.get('gad7_score', 0) > 9), (df_processed.get('gad7_score', 0) > 5) ],
        'ckd': [ ((df_processed.get('history_diabetes', 0) == 1) & (df_processed.get('history_hypertension', 0) == 1) & (df_processed['creatinine'] > 1.5)), ((df_processed.get('history_diabetes', 0) == 1) & (df_processed.get('history_hypertension', 0) == 1)), (df_processed.get('history_ckd', 0) == 1) ],
        'copd_exacerbation': [ ((df_processed.get('history_copd', 0) == 1) & (df_processed['respiratory_rate'] > 22)), ((df_processed.get('history_copd', 0) == 1) & (df_processed['respiratory_rate'] > 20)), ((df_processed.get('history_copd', 0) == 1) & (df_processed['respiratory_rate'] > 18)) ],
        'liver_disease': [ (df_processed['bilirubin'] > 2.5), (df_processed['bilirubin'] > 2.0), (df_processed['bilirubin'] > 1.8) ],
        'hypokalemia': [ (df_processed['potassium'] < 3.0), (df_processed['potassium'] < 3.5), (df_processed['potassium'] < 3.7) ],
        'hypernatremia': [ (df_processed['sodium'] > 150), (df_processed['sodium'] > 145), (df_processed['sodium'] > 142) ],
        'obesity': [ ((df_processed['weight_kg'] / safe_height_sq) > 35), ((df_processed['weight_kg'] / safe_height_sq) > 30), ((df_processed['weight_kg'] / safe_height_sq) > 28) ],
        'dehydration': [ ((df_processed['bun'] / safe_creatinine) > 25), ((df_processed['bun'] / safe_creatinine) > 20), (df_processed['sodium'] > 145) ],
        'thrombocytopenia': [ (df_processed['platelet_count'] < 100), (df_processed['platelet_count'] < 150), (df_processed['platelet_count'] < 180) ],
        'hyperkalemia': [ (df_processed['potassium'] > 5.5), (df_processed['potassium'] > 5.0), (df_processed['potassium'] > 4.8) ],
        'hyponatremia': [ (df_processed['sodium'] < 130), (df_processed['sodium'] < 135), (df_processed['sodium'] < 137) ],
        'leukopenia': [ (df_processed['wbc_count'] < 3.0), (df_processed['wbc_count'] < 4.0), (df_processed['wbc_count'] < 4.5) ],
    }

    timeframes = ['_6h', '_24h', '_48h']
    for name, rules in conditions.items():
        for i, time in enumerate(timeframes):
            df_processed[f'risk_{name}{time}'] = rules[i].astype(int)

    df_processed.fillna(0, inplace=True)
    df_processed = optimize_memory(df_processed)
    processed_chunks.append(df_processed)
    print(f"   -> Chunk processed. Features include instantaneous values and 3-hour trends.")

# --- COMBINE FINAL PROCESSED CHUNKS ---
print("\n--- Combining all processed chunks... ---")
df_processed = pd.concat(processed_chunks, ignore_index=True)
del processed_chunks
!rm -rf {temp_output_dir}
gc.collect()

print(f"\n✅ --- Final DataFrame with comprehensive features and labels created successfully! ---")
df_processed.info(verbose=False, show_counts=True)

import lightgbm as lgb
import gc
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

# --- Define feature and target columns ---
# This will automatically find all the new risk labels for all timeframes
target_diseases_multitime = [col for col in df_processed.columns if col.startswith('risk_')]
# All other columns (including new trend features) are used for prediction
feature_columns = [col for col in df_processed.columns if col not in target_diseases_multitime and col not in ['patient_id', 'obs_time']]
print(f"Training model with {len(feature_columns)} features to predict {len(target_diseases_multitime)} targets.")

# --- Split data for training ---
X_train, X_test, y_train, y_test = train_test_split(
    df_processed[feature_columns],
    df_processed[target_diseases_multitime],
    test_size=0.2,
    random_state=42
)

# --- Calculate and save feature medians for GUI imputation ---
print("--- Calculating and saving feature medians... ---")
feature_medians = df_processed[feature_columns].median().to_dict()
with open('feature_medians.json', 'w') as f:
    json.dump(feature_medians, f)
print("✅ Feature medians saved.")

# --- Clean up large DataFrame to free up RAM before training ---
del df_processed
gc.collect()

# --- Initialize and Train Model ---
lgb_estimator = lgb.LGBMClassifier(objective='binary', random_state=42, class_weight='balanced')
multi_target_model_multitime = MultiOutputClassifier(lgb_estimator, n_jobs=-1)

print("\n--- Training multi-timeframe predictive model (this may take a while)... ---")
multi_target_model_multitime.fit(X_train, y_train)
print("✅ Multi-timeframe model training complete!\n")

# --- Save artifacts for the GUI ---
print("\n--- Saving model and helper files for GUI... ---")
joblib.dump(multi_target_model_multitime, 'patient_risk_model_multitime.joblib')
with open('feature_columns_multitime.json', 'w') as f:
    json.dump(feature_columns, f)

# Create a clean list of base disease names for the GUI display
base_diseases = sorted(list(set([c.replace('risk_','').split('_')[0] for c in target_diseases_multitime])))
with open('target_diseases_base.json', 'w') as f:
    json.dump(base_diseases, f)
print("✅ All necessary files for the GUI have been saved!")

import pandas as pd
import numpy as np

# --- 1. Create Data for a New Hypothetical Patient ---
new_patient_data = pd.DataFrame({
    'age': [72], 'gender': ['male'],
    'history_hypertension': [1], 'history_diabetes': [1], 'history_chf': [1], 'history_copd': [1], 'history_anemia': [0],
    'history_hyperlipidemia': [1], 'history_cad': [1], 'history_ckd': [1], 'history_asthma': [0],
    'sbp': [160], 'dbp': [95], 'heart_rate': [95], 'respiratory_rate': [24],
    'pain_score': [6], 'height_m': [1.68], 'weight_kg': [80],
    'wbc_count': [11], 'hemoglobin': [12], 'hematocrit': [37], 'platelet_count': [210],
    'cholesterol_total': [230], 'triglycerides': [160], 'cholesterol_ldl': [150], 'cholesterol_hdl': [35],
    'hba1c': [7.0], 'phq2_score': [1], 'gad7_score': [10], 'temperature': [37.2], 'creatinine': [1.4], 'bun': [22],
    'sodium': [140], 'potassium': [4.2], 'glucose': [130], 'bilirubin': [0.8],
    'on_statin': [1], 'on_metformin': [1], 'on_aspirin': [1], 'on_lisinopril': [1],
    'history_cabg': [0], 'history_appendectomy': [0]
})

# --- 2. PREPARE DATA FOR PREDICTION ---
new_patient_processed = new_patient_data.copy()

# a. Create trend features (for a single snapshot, we assume stability)
for vital in ['heart_rate', 'sbp', 'dbp', 'respiratory_rate', 'temperature', 'glucose', 'wbc_count', 'hemoglobin']:
    if vital in new_patient_processed.columns:
        current_val = new_patient_processed[vital].iloc[0]
        new_patient_processed[f'{vital}_avg_3hr'] = current_val
        new_patient_processed[f'{vital}_std_3hr'] = 0.0

# b. Create other derived features
new_patient_processed = pd.get_dummies(new_patient_processed, columns=['gender'], drop_first=True, dtype=int)
new_patient_processed['map'] = ((2 * new_patient_processed['dbp']) + new_patient_processed['sbp']) / 3
new_patient_processed['shock_index'] = new_patient_processed['heart_rate'] / new_patient_processed['sbp'].replace(0, np.nan)
for col in ['sbp', 'dbp', 'heart_rate', 'wbc_count', 'hemoglobin', 'map']:
    new_patient_processed[f'delta_{col}'] = 0

# c. Align columns with the training set
# This uses the 'feature_columns' variable from the training block
new_patient_processed = new_patient_processed.reindex(columns=feature_columns, fill_value=0)

# --- 3. GET PREDICTIONS ---
print("--- Making predictions with the multi-timeframe model... ---")
probabilities = multi_target_model_multitime.predict_proba(new_patient_processed)

# V-- THIS IS THE CORRECTED LINE --V
# Use 'target_diseases_multitime' (from the training block) as the keys, not 'model.classes_'
results_map = {disease: prob[0, 1] for disease, prob in zip(target_diseases_multitime, probabilities)}

# --- 4. DISPLAY RESULTS IN A TABLE ---
output_df_data = []
# This uses the 'base_diseases' list from the training block
for disease_base in base_diseases:
    clean_name = disease_base.replace('_',' ').title()
    output_df_data.append({
        "Condition": clean_name,
        "6-Hour Risk": f"{results_map.get('risk_' + disease_base + '_6h', 0):.2%}",
        "24-Hour Risk": f"{results_map.get('risk_' + disease_base + '_24h', 0):.2%}",
        "48-Hour Risk": f"{results_map.get('risk_' + disease_base + '_48h', 0):.2%}",
        "_sort_key": results_map.get('risk_' + disease_base + '_24h', 0)
    })

# Sort by 24-hour risk and display as a DataFrame
sorted_results = sorted(output_df_data, key=lambda x: x["_sort_key"], reverse=True)
results_df = pd.DataFrame(sorted_results).drop(columns=['_sort_key'])

print("\n--- Multi-Timeframe Predictive Risk Assessment ---")
print(results_df.to_string())

from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
import gc
import pandas as pd

print("--- Evaluating Multi-Timeframe Model Performance on Unseen Test Data ---\n")

# --- Use the existing model and test set from the training block to get predictions ---
print("Generating predictions on the test set...")
y_pred_proba = multi_target_model_multitime.predict_proba(X_test)
y_pred_binary = multi_target_model_multitime.predict(X_test)

# --- Delete large DataFrames to free up RAM before calculations ---
print("Deleting large train/test sets to free up RAM...")
del X_train, X_test, y_train
gc.collect()

# --- Evaluate each condition and timeframe ---
auc_scores = []
# Get the full list of target labels (e.g., 'risk_sepsis_6h', 'risk_sepsis_24h', etc.)
target_labels = y_test.columns

for i, disease_full_name in enumerate(target_labels):
    true_labels = y_test[disease_full_name]
    predicted_probs = y_pred_proba[i][:, 1]
    predicted_labels = y_pred_binary[:, i]

    # Calculate AUC, handling cases with only one class
    if len(np.unique(true_labels)) > 1:
        auc_score = roc_auc_score(true_labels, predicted_probs)
        auc_scores.append(auc_score)
        auc_text = f"AUC-ROC Score: {auc_score:.4f}"
    else:
        auc_text = "AUC-ROC Score: N/A (only one class present in test set)"

    # Create a cleaner name for the report title, e.g., "Sepsis (24h)"
    parts = disease_full_name.replace('risk_', '').split('_')
    clean_name = parts[0].replace('_',' ').title()
    timeframe = parts[-1]
    report_title = f"{clean_name} ({timeframe})"

    print(f"--- Performance for: {report_title} ---")
    print(auc_text)
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, zero_division=0))
    print("-" * 50 + "\n")

if auc_scores:
    print(f"✅ Average AUC-ROC across {len(auc_scores)} conditions and timeframes: {np.mean(auc_scores):.4f}")

# --- 1. INSTALL GRADIO ---
!pip install gradio -q
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import json

# --- 2. LOAD SAVED ARTIFACTS ---
print("--- Loading multi-timeframe model and helper files... ---")
try:
    model = joblib.load('patient_risk_model_multitime.joblib')
    feature_columns = json.load(open('feature_columns_multitime.json'))
    base_diseases = json.load(open('target_diseases_base.json'))
    feature_medians = json.load(open('feature_medians.json'))
    print("✅ Model and all necessary files loaded successfully.")
except FileNotFoundError as e:
    print(f"--- ERROR: Could not find a required file: {e.filename} ---")
    model, feature_columns, base_diseases, feature_medians = None, [], [], {}

# --- WORKAROUND: Reconstruct the full list of target labels ---
# This uses the 'base_diseases' list to create the full, ordered list of target names
# that the model was trained on. This is crucial for correctly mapping predictions.
target_diseases_multitime = []
if base_diseases:
    for disease in base_diseases:
        # Assuming the model was trained on 6h, 24h, and 48h targets for each base disease
        target_diseases_multitime.append(f"risk_{disease}_6h")
        target_diseases_multitime.append(f"risk_{disease}_24h")
        target_diseases_multitime.append(f"risk_{disease}_48h")
    print(f"Reconstructed {len(target_diseases_multitime)} target labels for mapping predictions.")


# --- 3. CREATE THE PREDICTION FUNCTION ---
def predict_risk_multitime(ignore_features, *args):
    if model is None: return pd.DataFrame({"Error": ["Model not loaded. Please train the model first."]})

    # Map all the Gradio inputs back to a dictionary
    input_names = [inp.label for inp in all_inputs if inp.label and inp.label != "Ignore Inputs"]
    input_values = dict(zip(input_names, args))

    input_data = {}
    for name, value in input_values.items():
        key = name.lower().replace(' ', '_').split('(')[0].strip()
        if key == 'systolic_bp': key = 'sbp'
        if key == 'diastolic_bp': key = 'dbp'

        # Use median if the field is blank (None) OR if it was checked in the 'Ignore' box
        if value is None or name in ignore_features:
            input_data[key] = feature_medians.get(key)
        else:
            input_data[key] = value

    # Handle checkboxes for history and medications
    history_map = { 'Hypertension': 'history_hypertension', 'Diabetes': 'history_diabetes', 'CHF': 'history_chf', 'COPD': 'history_copd', 'Anemia': 'history_anemia', 'Hyperlipidemia': 'history_hyperlipidemia', 'CAD': 'history_cad', 'CKD': 'history_ckd', 'Asthma': 'history_asthma' }
    for choice, key in history_map.items():
        input_data[key] = 1 if choice in args[-3] else 0 # medical_history
    input_data['history_cabg'] = 1 if 'CABG' in args[-2] else 0 # surgical_history
    input_data['on_statin'] = 1 if 'Statin' in args[-1] else 0 # current_meds
    input_data['on_metformin'] = 1 if 'Metformin' in args[-1] else 0

    # Impute trend features (assume stability for snapshot prediction)
    for vital in ['heart_rate', 'sbp', 'dbp', 'respiratory_rate', 'temperature', 'glucose', 'wbc_count', 'hemoglobin']:
        current_val = input_data.get(vital, feature_medians.get(vital, 0))
        input_data[f'{vital}_avg_3hr'] = current_val
        input_data[f'{vital}_std_3hr'] = 0.0

    # Preprocess and Align the DataFrame
    df = pd.DataFrame([input_data])
    if 'gender' in df.columns: df = pd.get_dummies(df, columns=['gender'], drop_first=True)
    df['map'] = ((2 * df['dbp']) + df['sbp']) / 3
    df['shock_index'] = df['heart_rate'] / df['sbp'].replace(0, np.nan)
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Make Predictions
    probabilities = model.predict_proba(df)

    # Use the reconstructed list of target names to map the results
    results_map = {disease: prob[0, 1] for disease, prob in zip(target_diseases_multitime, probabilities)}

    # Format the output table
    output_df_data = []
    for disease_base in base_diseases:
        clean_name = disease_base.replace('_',' ').title()
        output_df_data.append({
            "Condition": clean_name,
            "6-Hour Risk": f"{results_map.get('risk_' + disease_base + '_6h', 0):.2%}",
            "24-Hour Risk": f"{results_map.get('risk_' + disease_base + '_24h', 0):.2%}",
            "48-Hour Risk": f"{results_map.get('risk_' + disease_base + '_48h', 0):.2%}",
            "_sort_key": results_map.get('risk_' + disease_base + '_24h', 0)
        })
    sorted_results = sorted(output_df_data, key=lambda x: x["_sort_key"], reverse=True)
    return pd.DataFrame(sorted_results).drop(columns=['_sort_key'])

# --- 4. DEFINE THE GRADIO INTERFACE ---
numerical_inputs_labels = [ "Age", "Systolic BP", "Diastolic BP", "Heart Rate", "Respiratory Rate", "Temperature (°C)", "Pain Score (0-10)", "Height (m)", "Weight (kg)", "WBC Count", "Hemoglobin", "Hematocrit", "Platelet Count", "Total Cholesterol", "Triglycerides", "LDL Cholesterol", "HDL Cholesterol", "Glucose", "Creatinine", "BUN", "Sodium", "Potassium", "Bilirubin", "HbA1c", "PHQ-2 Score", "GAD-7 Score" ]
with gr.Blocks() as demo:
    gr.Markdown("# Patient Multi-Timeframe Risk Prediction")
    gr.Markdown("Enter patient data. Check a box in 'Ignore Inputs' to use a typical value for any field you don't have data for.")
    with gr.Row():
        ignore_box = gr.CheckboxGroup(choices=numerical_inputs_labels, label="Ignore Inputs", scale=1)
        with gr.Column(scale=2):
            with gr.Row():
                age = gr.Number(label="Age")
                gender = gr.Radio(label="Gender", choices=["male", "female"], value="male")
            with gr.Accordion("Vitals & Measurements", open=False):
                 sbp, dbp, heart_rate = gr.Number(label="Systolic BP"), gr.Number(label="Diastolic BP"), gr.Number(label="Heart Rate")
                 respiratory_rate, temperature, pain_score = gr.Number(label="Respiratory Rate"), gr.Number(label="Temperature (°C)"), gr.Number(label="Pain Score (0-10)")
                 height_m, weight_kg = gr.Number(label="Height (m)"), gr.Number(label="Weight (kg)")
            with gr.Accordion("Labs", open=False):
                wbc_count, hemoglobin, hematocrit, platelet_count = gr.Number(label="WBC Count"), gr.Number(label="Hemoglobin"), gr.Number(label="Hematocrit"), gr.Number(label="Platelet Count")
                cholesterol_total, triglycerides, cholesterol_ldl, cholesterol_hdl = gr.Number(label="Total Cholesterol"), gr.Number(label="Triglycerides"), gr.Number(label="LDL Cholesterol"), gr.Number(label="HDL Cholesterol")
                glucose, creatinine, bun, sodium = gr.Number(label="Glucose"), gr.Number(label="Creatinine"), gr.Number(label="BUN"), gr.Number(label="Sodium")
                potassium, bilirubin, hba1c = gr.Number(label="Potassium"), gr.Number(label="Bilirubin"), gr.Number(label="HbA1c")
            with gr.Accordion("Scores & History", open=False):
                phq2_score = gr.Number(label="PHQ-2 Score")
                gad7_score = gr.Number(label="GAD-7 Score")
                medical_history = gr.CheckboxGroup(label="Medical History", choices=["Hypertension", "Diabetes", "CHF", "COPD", "Anemia", "Hyperlipidemia", "CAD", "CKD", "Asthma"])
                surgical_history = gr.CheckboxGroup(label="Surgical History", choices=["CABG"])
                current_meds = gr.CheckboxGroup(label="Current Medications", choices=["Statin", "Metformin"])
    submit_btn = gr.Button("Predict Risk")
    output = gr.Dataframe(headers=["Condition", "6-Hour Risk", "24-Hour Risk", "48-Hour Risk"], label="Predicted Condition Risks", wrap=True)
    all_inputs = [ignore_box, age, gender, sbp, dbp, heart_rate, respiratory_rate, temperature, pain_score, height_m, weight_kg, wbc_count, hemoglobin, hematocrit, platelet_count, cholesterol_total, triglycerides, cholesterol_ldl, cholesterol_hdl, glucose, creatinine, bun, sodium, potassium, bilirubin, hba1c, phq2_score, gad7_score, medical_history, surgical_history, current_meds]
    submit_btn.click(fn=predict_risk_multitime, inputs=all_inputs, outputs=output)

# --- 5. LAUNCH THE GUI ---
demo.launch(share=True, debug=True)
