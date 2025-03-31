import pickle
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# Define reference ranges for diabetes parameters
DIABETES_RANGES = {
    "Pregnancies": {"min": 0, "max": 17, "info": "Number of pregnancies (0-17)"},
    "Glucose": {"min": 70, "max": 200, "info": "Fasting blood glucose level (mg/dL). Normal: 70-99, Prediabetes: 100-125, Diabetes: >126"},
    "BloodPressure": {"min": 60, "max": 140, "info": "Diastolic blood pressure (mm Hg). Normal: <80, Elevated: 80-89, High: ≥90"},
    "SkinThickness": {"min": 10, "max": 50, "info": "Triceps skin fold thickness (mm). Normal range: 10-25"},
    "Insulin": {"min": 0, "max": 846, "info": "2-Hour serum insulin (mu U/ml). Normal: <140, Prediabetes: 140-199, Diabetes: ≥200"},
    "BMI": {"min": 18.5, "max": 50, "info": "Body mass index (kg/m²). Underweight: <18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: ≥30"},
    "DiabetesPedigreeFunction": {"min": 0.078, "max": 2.42, "info": "Diabetes pedigree function (genetic score). Higher values indicate stronger family history"},
    "Age": {"min": 21, "max": 90, "info": "Age in years"}
}

# Enhanced reference ranges for heart disease parameters with better descriptions
HEART_RANGES = {
    "age": {"min": 25, "max": 90, "info": "Age in years. Risk increases with age (men >45, women >55)"},
    "sex": {"min": 0, "max": 1, "info": "Sex (0 = female, 1 = male). Men are generally at higher risk"},
    "cp": {
        "min": 0, 
        "max": 3, 
        "info": "Chest pain type:\n"
                "0 = Typical angina (chest pain related to heart)\n"
                "1 = Atypical angina (non-heart related chest pain)\n"
                "2 = Non-anginal pain (not related to heart)\n"
                "3 = Asymptomatic (no symptoms)"
    },
    "trestbps": {
        "min": 90, 
        "max": 200, 
        "info": "Resting blood pressure (mm Hg):\n"
                "Normal: <120\n"
                "Elevated: 120-129\n"
                "Stage 1 HTN: 130-139\n"
                "Stage 2 HTN: ≥140"
    },
    "chol": {
        "min": 100, 
        "max": 500, 
        "info": "Serum cholesterol (mg/dl):\n"
                "Desirable: <200\n"
                "Borderline high: 200-239\n"
                "High: ≥240"
    },
    "fbs": {
        "min": 0, 
        "max": 1, 
        "info": "Fasting blood sugar >120 mg/dl:\n"
                "0 = False (normal)\n"
                "1 = True (elevated)"
    },
    "restecg": {
        "min": 0, 
        "max": 2, 
        "info": "Resting ECG results:\n"
                "0 = Normal\n"
                "1 = ST-T wave abnormality (may indicate ischemia)\n"
                "2 = Left ventricular hypertrophy (thickened heart wall)"
    },
    "thalach": {
        "min": 70, 
        "max": 220, 
        "info": "Maximum heart rate achieved during exercise.\n"
                "Age-predicted maximum is roughly 220 - age.\n"
                "Lower values may indicate reduced heart function."
    },
    "exang": {
        "min": 0, 
        "max": 1, 
        "info": "Exercise induced angina:\n"
                "0 = No chest pain during exercise\n"
                "1 = Chest pain during exercise"
    },
    "oldpeak": {
        "min": 0, 
        "max": 6.2, 
        "info": "ST depression induced by exercise relative to rest.\n"
                "Measures heart stress during exercise.\n"
                "Higher values indicate more ischemia (lack of blood flow to heart)."
    },
    "slope": {
        "min": 0, 
        "max": 2, 
        "info": "Slope of peak exercise ST segment:\n"
                "0 = Upsloping (normal)\n"
                "1 = Flat (may indicate ischemia)\n"
                "2 = Downsloping (may indicate coronary artery disease)"
    },
    "ca": {
        "min": 0, 
        "max": 4, 
        "info": "Number of major vessels (0-4) visible on fluoroscopy.\n"
                "Higher numbers indicate more blocked vessels."
    },
    "thal": {
        "min": 0, 
        "max": 3, 
        "info": "Thalassemia (blood disorder affecting hemoglobin):\n"
                "0 = Normal\n"
                "1 = Fixed defect (scar tissue from prior damage)\n"
                "2 = Reversible defect (area with reduced blood flow)\n"
                "3 = Unreported/unknown"
    }
}

# ASCII Art for the application
ASCII_ART = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ███╗   ███╗███████╗██████╗ ██╗ ██████╗ █████╗ ██╗         ███████╗     ║
║   ████╗ ████║██╔════╝██╔══██╗██║██╔════╝██╔══██╗██║         ██╔════╝     ║
║   ██╔████╔██║█████╗  ██║  ██║██║██║     ███████║██║         ███████╗     ║
║   ██║╚██╔╝██║██╔══╝  ██║  ██║██║██║     ██╔══██║██║         ╚════██║     ║
║   ██║ ╚═╝ ██║███████╗██████╔╝██║╚██████╗██║  ██║███████╗    ███████║     ║
║   ╚═╝     ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝    ╚══════╝     ║
║                                                                           ║
║                             SHIELD                                        ║
║                                                                           ║
║            ML-Powered Disease Prediction & Prevention System              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

def print_header():
    """Print the application header with ASCII art"""
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear the screen
    print(ASCII_ART)
    print("\n" + "=" * 80)
    print("Welcome to Medical Shield - Advanced Disease Prediction System".center(80))
    print("=" * 80)

def print_loading(message):
    """Print a loading animation with a message"""
    for i in range(3):
        print(f"\r{message}" + "." * (i + 1) + " " * (5 - i), end="", flush=True)
        time.sleep(0.3)
    print("\r" + " " * 80, end="", flush=True)

def load_data():
    """Load and process the datasets"""
    print_loading("Loading datasets")
    
    # Load dataset
    try:
        diabetes_df = pd.read_csv(r"Diabetes_prediction.csv")
        heart_df = pd.read_csv(r"HeartDiseaseTrain-Test.csv")
    except FileNotFoundError:
        print("\n❌ Error: Dataset files not found. Please check the file paths.")
        exit(1)
    
    print("✅ Datasets loaded successfully")
    
    # Print dataset information
    print("\n📊 Dataset Information:")
    print(f"  • Diabetes dataset: {diabetes_df.shape[0]} samples, {diabetes_df.shape[1]} features")
    print(f"  • Heart disease dataset: {heart_df.shape[0]} samples, {heart_df.shape[1]} features")
    
    print_loading("Processing datasets")
    
    # Process datasets
    diabetes_df, heart_df = process_datasets(diabetes_df, heart_df)
    
    print("✅ Data processing complete")
    
    return diabetes_df, heart_df

def process_datasets(diabetes_df, heart_df):
    """Process and clean the datasets"""
    # Check for columns with all NaN values
    diabetes_null_cols = [col for col in diabetes_df.columns if diabetes_df[col].isna().all()]
    heart_null_cols = [col for col in heart_df.columns if heart_df[col].isna().all()]

    if diabetes_null_cols:
        print(f"\n⚠️ Warning: Dropping columns with all missing values in diabetes dataset: {diabetes_null_cols}")
        diabetes_df = diabetes_df.drop(columns=diabetes_null_cols)

    if heart_null_cols:
        print(f"\n⚠️ Warning: Dropping columns with all missing values in heart dataset: {heart_null_cols}")
        heart_df = heart_df.drop(columns=heart_null_cols)

    # Convert categorical variables to numeric
    diabetes_cat_cols = diabetes_df.select_dtypes(include=['object']).columns
    heart_cat_cols = heart_df.select_dtypes(include=['object']).columns

    # Handle categorical columns
    for df, cat_cols in [(diabetes_df, diabetes_cat_cols), (heart_df, heart_cat_cols)]:
        for col in cat_cols:
            if col != "Diagnosis" and col != "target":  # Don't convert target variables yet
                df[col] = pd.Categorical(df[col]).codes
                df[col] = df[col].replace(-1, np.nan)

    # Convert remaining columns to numeric
    for col in diabetes_df.columns:
        if col not in diabetes_cat_cols:
            diabetes_df[col] = pd.to_numeric(diabetes_df[col], errors='coerce')

    for col in heart_df.columns:
        if col not in heart_cat_cols:
            heart_df[col] = pd.to_numeric(heart_df[col], errors='coerce')

    # Handle missing values
    for col in diabetes_df.columns:
        if col != "Diagnosis" and diabetes_df[col].isna().any():
            median_val = diabetes_df[col].median()
            if not pd.isna(median_val):
                diabetes_df[col] = diabetes_df[col].fillna(median_val)
            else:
                mode_val = diabetes_df[col].mode()
                if not mode_val.empty:
                    diabetes_df[col] = diabetes_df[col].fillna(mode_val[0])
                else:
                    print(f"\n⚠️ Warning: Dropping column {col} due to excessive missing values")
                    diabetes_df = diabetes_df.drop(columns=[col])

    for col in heart_df.columns:
        if col != "target" and heart_df[col].isna().any():
            median_val = heart_df[col].median()
            if not pd.isna(median_val):
                heart_df[col] = heart_df[col].fillna(median_val)
            else:
                mode_val = heart_df[col].mode()
                if not mode_val.empty:
                    heart_df[col] = heart_df[col].fillna(mode_val[0])
                else:
                    print(f"\n⚠️ Warning: Dropping column {col} due to excessive missing values")
                    heart_df = heart_df.drop(columns=[col])

    # Check for remaining NaN values
    df_nan_check_diabetes = diabetes_df.isna().sum().sum()
    df_nan_check_heart = heart_df.isna().sum().sum()

    if df_nan_check_diabetes > 0:
        print(f"\n⚠️ Warning: Dropping {diabetes_df.isna().sum().sum()} rows with missing values in diabetes dataset")
        diabetes_df = diabetes_df.dropna()

    if df_nan_check_heart > 0:
        print(f"\n⚠️ Warning: Dropping {heart_df.isna().sum().sum()} rows with missing values in heart dataset")
        heart_df = heart_df.dropna()

    # Verify no NaN values remain
    df_nan_check = diabetes_df.isna().sum().sum() + heart_df.isna().sum().sum()
    if df_nan_check > 0:
        print("\n❌ Error: Unable to handle all missing values. Please check the datasets manually.")
        exit(1)

    # Check for target columns
    if "Diagnosis" not in diabetes_df.columns:
        print("\n❌ Error: 'Diagnosis' column not found in diabetes dataset!")
        exit(1)
    if "target" not in heart_df.columns:
        print("\n❌ Error: 'target' column not found in heart dataset!")
        exit(1)

    return diabetes_df, heart_df

def train_or_load_models(X_diabetes, y_diabetes, X_heart, y_heart):
    """Train new models or load existing ones"""
    # Check if model files exist
    if not all(os.path.exists(f) for f in ["random_forest_diabetes.pkl", "svm_heart.pkl", "scaler_diabetes.pkl", "scaler_heart.pkl"]):
        print("\n🔄 Training new models (this may take a few moments)...")
        
        # Create and train the models
        print_loading("Preprocessing data")
        
        # Split datasets
        X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)
        X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
        
        # Scale data
        scaler_d = StandardScaler()
        X_train_d_scaled = scaler_d.fit_transform(X_train_d)
        X_test_d_scaled = scaler_d.transform(X_test_d)
        
        scaler_h = StandardScaler()
        X_train_h_scaled = scaler_h.fit_transform(X_train_h)
        X_test_h_scaled = scaler_h.transform(X_test_h)
        
        # Train models
        print_loading("Training diabetes model")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_d_scaled, y_train_d)
        
        print_loading("Training heart disease model")
        svm = SVC(probability=True, random_state=42)
        svm.fit(X_train_h_scaled, y_train_h)
        
        # Save models
        print_loading("Saving models")
        with open("random_forest_diabetes.pkl", "wb") as f:
            pickle.dump(rf, f)
        
        with open("svm_heart.pkl", "wb") as f:
            pickle.dump(svm, f)
        
        with open("scaler_diabetes.pkl", "wb") as f:
            pickle.dump(scaler_d, f)
        
        with open("scaler_heart.pkl", "wb") as f:
            pickle.dump(scaler_h, f)
        
        print("✅ Models trained and saved successfully!")
        
        # Evaluate models
        y_pred_d = rf.predict(X_test_d_scaled)
        y_pred_h = svm.predict(X_test_h_scaled)
        
        print("\n📊 Initial Model Evaluation:")
        print(f"  • Diabetes Model Accuracy: {accuracy_score(y_test_d, y_pred_d):.2f}")
        print(f"  • Heart Disease Model Accuracy: {accuracy_score(y_test_h, y_pred_h):.2f}")
        
        return rf, svm, scaler_d, scaler_h
    
    else:
        print("\n🔄 Loading existing models...")
        
        # Load models
        with open("random_forest_diabetes.pkl", "rb") as f:
            rf = pickle.load(f)
        
        with open("svm_heart.pkl", "rb") as f:
            svm = pickle.load(f)
        
        with open("scaler_diabetes.pkl", "rb") as f:
            scaler_d = pickle.load(f)
        
        with open("scaler_heart.pkl", "rb") as f:
            scaler_h = pickle.load(f)
        
        print("✅ Models loaded successfully!")
        
        return rf, svm, scaler_d, scaler_h

def get_input_with_range(prompt, min_val, max_val, info=""):
    """Get input from user with range validation"""
    while True:
        print(f"\n{prompt} [{min_val}-{max_val}]")
        if info:
            print(f"ℹ️  {info}")
        try:
            value = float(input("Enter value: "))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"❌ Value must be between {min_val} and {max_val}. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

def get_categorical_input(prompt, options):
    """Get categorical input from user with description"""
    while True:
        print(f"\n{prompt}")
        for i, (value, description) in enumerate(options.items(), 1):
            print(f"{i}. {value} = {description}")
        try:
            choice = int(input("Enter choice number: "))
            if 1 <= choice <= len(options):
                return list(options.keys())[choice-1]
            else:
                print(f"❌ Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("❌ Please enter a valid number.")

def predict_diabetes(rf, scaler_d, X_diabetes):
    """Predict diabetes using the Random Forest model"""
    print("\n" + "=" * 80)
    print("Diabetes Risk Assessment".center(80))
    print("=" * 80)
    
    print("\nPlease enter the following patient details:")
    
    features = []
    
    for i, col in enumerate(X_diabetes.columns):
        if col in DIABETES_RANGES:
            value = get_input_with_range(
                f"{i+1}. {col}",
                DIABETES_RANGES[col]["min"],
                DIABETES_RANGES[col]["max"],
                DIABETES_RANGES[col]["info"]
            )
            features.append(value)
        else:
            # Fallback for columns not in our range dictionary
            while True:
                try:
                    print(f"\n{i+1}. {col}")
                    value = float(input("Enter value: "))
                    features.append(value)
                    break
                except ValueError:
                    print("❌ Please enter a valid number.")
    
    print("\n🔄 Analyzing patient data...")
    time.sleep(1)
    
    # Scale features and make prediction
    features_scaled = scaler_d.transform([features])
    prediction = rf.predict(features_scaled)[0]
    probability = rf.predict_proba(features_scaled)[0][1]
    
    print("\n" + "=" * 80)
    print("Diabetes Assessment Results".center(80))
    print("=" * 80)
    
    if prediction == 1:
        print("\n🚨 Assessment: High risk of diabetes detected")
        print(f"Confidence: {probability:.2f} (or {probability*100:.1f}%)")
        
        if probability > 0.8:
            risk_level = "Very High"
        elif probability > 0.6:
            risk_level = "High"
        else:
            risk_level = "Moderate"
            
        print(f"Risk Level: {risk_level}")
        
        print("\nRecommendations:")
        print("• Consult with a healthcare provider as soon as possible")
        print("• Consider fasting blood glucose and HbA1c tests")
        print("• Monitor blood glucose levels regularly")
        print("• Maintain a healthy diet and exercise routine")
        print("• Limit intake of sugary foods and beverages")
    else:
        print("\n✅ Assessment: Low risk of diabetes")
        print(f"Confidence: {1-probability:.2f} (or {(1-probability)*100:.1f}%)")
        
        print("\nRecommendations:")
        print("• Continue regular health check-ups")
        print("• Maintain a healthy diet rich in vegetables and whole grains")
        print("• Regular physical activity (at least 150 minutes per week)")
        print("• Maintain a healthy weight")
        print("• Limit alcohol consumption")
    
    print("\nNote: This assessment is for informational purposes only and should not replace")
    print("professional medical advice. Always consult with a healthcare provider for diagnosis.")
    
    input("\nPress Enter to continue...")

def predict_heart_disease(svm, scaler_h, X_heart):
    """Predict heart disease using the SVM model"""
    print("\n" + "=" * 80)
    print("Heart Disease Risk Assessment".center(80))
    print("=" * 80)
    
    print("\nPlease enter the following patient details:")
    
    features = []
    
    for i, col in enumerate(X_heart.columns):
        if col in HEART_RANGES:
            # Handle categorical features differently
            if col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]:
                if col == "sex":
                    options = {0: "Female", 1: "Male"}
                elif col == "cp":
                    options = {
                        0: "Typical angina (chest pain related to heart)",
                        1: "Atypical angina (non-heart related chest pain)",
                        2: "Non-anginal pain (not related to heart)",
                        3: "Asymptomatic (no symptoms)"
                    }
                elif col == "fbs":
                    options = {0: "Fasting blood sugar ≤120 mg/dl (normal)", 
                               1: "Fasting blood sugar >120 mg/dl (elevated)"}
                elif col == "restecg":
                    options = {
                        0: "Normal ECG", 
                        1: "ST-T wave abnormality (may indicate ischemia)",
                        2: "Left ventricular hypertrophy (thickened heart wall)"
                    }
                elif col == "exang":
                    options = {0: "No exercise-induced angina", 1: "Exercise-induced angina present"}
                elif col == "slope":
                    options = {
                        0: "Upsloping (normal)", 
                        1: "Flat (may indicate ischemia)",
                        2: "Downsloping (may indicate coronary artery disease)"
                    }
                elif col == "thal":
                    options = {
                        0: "Normal", 
                        1: "Fixed defect (scar tissue from prior damage)",
                        2: "Reversible defect (area with reduced blood flow)",
                        3: "Unreported/unknown"
                    }
                
                value = get_categorical_input(
                    f"{i+1}. {col} - Select option:",
                    options
                )
                features.append(value)
            else:
                value = get_input_with_range(
                    f"{i+1}. {col}",
                    HEART_RANGES[col]["min"],
                    HEART_RANGES[col]["max"],
                    HEART_RANGES[col]["info"]
                )
                features.append(value)
        else:
            # Fallback for columns not in our range dictionary
            while True:
                try:
                    print(f"\n{i+1}. {col}")
                    value = float(input("Enter value: "))
                    features.append(value)
                    break
                except ValueError:
                    print("❌ Please enter a valid number.")
    
    print("\n🔄 Analyzing patient data...")
    time.sleep(1)
    
    # Scale features and make prediction
    features_scaled = scaler_h.transform([features])
    prediction = svm.predict(features_scaled)[0]
    probability = svm.predict_proba(features_scaled)[0][1]
    
    print("\n" + "=" * 80)
    print("Heart Disease Assessment Results".center(80))
    print("=" * 80)
    
    if prediction == 1:
        print("\n🚨 Assessment: High risk of heart disease detected")
        print(f"Confidence: {probability:.2f} (or {probability*100:.1f}%)")
        
        if probability > 0.8:
            risk_level = "Very High"
            recommendation = "• Seek immediate medical attention from a cardiologist"
        elif probability > 0.6:
            risk_level = "High"
            recommendation = "• Schedule an appointment with a cardiologist soon"
        else:
            risk_level = "Moderate"
            recommendation = "• Consult with your primary care physician"
            
        print(f"Risk Level: {risk_level}")
        
        print("\nRecommendations:")
        print(recommendation)
        print("• Consider diagnostic tests like stress test, echocardiogram, or angiogram")
        print("• Monitor blood pressure and cholesterol levels regularly")
        print("• Adopt a heart-healthy diet (low in saturated fats and sodium)")
        print("• Engage in regular cardiovascular exercise as recommended by your doctor")
        print("• Quit smoking and limit alcohol intake")
        print("• Manage stress through relaxation techniques")
        
        # Additional specific recommendations based on input features
        if features[2] in [1, 2, 3]:  # If any chest pain reported
            print("\n⚠️ Special Note: Patient reported chest pain - consider immediate evaluation")
        
        if features[8] == 1:  # Exercise induced angina
            print("⚠️ Special Note: Exercise-induced angina present - limit strenuous activity until evaluated")
        
        if features[3] > 140:  # High blood pressure
            print("⚠️ Special Note: Elevated blood pressure - monitor regularly and consider treatment")
        
        if features[4] > 240:  # High cholesterol
            print("⚠️ Special Note: High cholesterol levels detected - dietary changes may be needed")
    else:
        print("\n✅ Assessment: Low risk of heart disease")
        print(f"Confidence: {1-probability:.2f} (or {(1-probability)*100:.1f}%)")
        
        print("\nRecommendations:")
        print("• Continue regular health check-ups and cardiac screenings")
        print("• Maintain a heart-healthy diet (fruits, vegetables, whole grains)")
        print("• Regular physical activity (at least 150 minutes per week)")
        print("• Maintain healthy weight and blood pressure")
        print("• Avoid smoking and limit alcohol consumption")
        print("• Manage stress through relaxation techniques or mindfulness")
    
    print("\nNote: This assessment is for informational purposes only and should not replace")
    print("professional medical advice. Always consult with a healthcare provider for diagnosis.")
    
    input("\nPress Enter to continue...")

def evaluate_models(rf, svm, scaler_d, scaler_h, X_diabetes, y_diabetes, X_heart, y_heart):
    """Evaluate and display model performance"""
    print("\n" + "=" * 80)
    print("Model Performance Evaluation".center(80))
    print("=" * 80)
    
    print("\n🔄 Evaluating diabetes prediction model...")
    
    # Split data for evaluation
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
    
    # Scale test data
    X_test_d_scaled = scaler_d.transform(X_test_d)
    X_test_h_scaled = scaler_h.transform(X_test_h)
    
    # Evaluate models
    y_pred_d = rf.predict(X_test_d_scaled)
    y_pred_h = svm.predict(X_test_h_scaled)
    
    # Calculate metrics
    acc_d = accuracy_score(y_test_d, y_pred_d)
    acc_h = accuracy_score(y_test_h, y_pred_h)
    
    print("\n📊 Diabetes Model Performance:")
    print(f"  • Accuracy: {acc_d:.2f} ({acc_d*100:.1f}%)")
    print(f"  • Test dataset size: {len(X_test_d)} samples")
    print("\nClassification Report:")
    print(classification_report(y_test_d, y_pred_d))
    
    print("\n🔄 Evaluating heart disease prediction model...")
    
    print("\n📊 Heart Disease Model Performance:")
    print(f"  • Accuracy: {acc_h:.2f} ({acc_h*100:.1f}%)")
    print(f"  • Test dataset size: {len(X_test_h)} samples")
    print("\nClassification Report:")
    print(classification_report(y_test_h, y_pred_h))
    
    print("\nNote: Model performance may vary based on the quality and quantity of training data.")
    print("Regular retraining with new data can improve prediction accuracy.")
    
    input("\nPress Enter to continue...")

def display_about():
    """Display information about the application"""
    print("\n" + "=" * 80)
    print("About Medical Shield".center(80))
    print("=" * 80)
    
    print("""
Medical Shield is an advanced machine learning-powered disease prediction system
designed to assist healthcare professionals in early detection and risk assessment
of diabetes and heart disease.

Key Features:
• Diabetes risk assessment using Random Forest algorithm
• Heart disease prediction using Support Vector Machine (SVM)
• Detailed patient assessment with confidence levels
• Personalized recommendations based on risk profiles
• Comprehensive model evaluation metrics

This application is for educational and informational purposes only. It should not
replace professional medical advice, diagnosis, or treatment. Always consult with
a qualified healthcare provider for medical concerns.

Developed for hackathon demonstration purposes.
""")
    
    input("\nPress Enter to continue...")

def main():
    """Main function to run the application"""
    print_header()
    
    print("\n🔄 Initializing Medical Shield...")
    time.sleep(1)
    
    # Load and process data
    diabetes_df, heart_df = load_data()
    
    # Separate features and target
    X_diabetes = diabetes_df.drop(columns=["Diagnosis"])
    y_diabetes = diabetes_df["Diagnosis"]
    
    X_heart = heart_df.drop(columns=["target"])
    y_heart = heart_df["target"]
    
    # Train or load models
    rf, svm, scaler_d, scaler_h = train_or_load_models(X_diabetes, y_diabetes, X_heart, y_heart)
    
    # Main application loop
    while True:
        print_header()
        
        print("\nWhat would you like to do?")
        print("1️⃣  Diabetes Risk Assessment")
        print("2️⃣  Heart Disease Risk Assessment")
        print("3️⃣  Evaluate Model Performance")
        print("4️⃣  About Medical Shield")
        print("5️⃣  Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            predict_diabetes(rf, scaler_d, X_diabetes)
        elif choice == "2":
            predict_heart_disease(svm, scaler_h, X_heart)
        elif choice == "3":
            evaluate_models(rf, svm, scaler_d, scaler_h, X_diabetes, y_diabetes, X_heart, y_heart)
        elif choice == "4":
            display_about()
        elif choice == "5":
            print("\n👋 Thank you for using Medical Shield. Stay healthy!")
            print("Exiting application...")
            time.sleep(1)
            break
        else:
            print("\n❌ Invalid choice! Please enter a number between 1 and 5.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application terminated by user. Stay healthy!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        print("Please report this issue to the development team.")