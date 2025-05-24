import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

def train_model():
    try:
        # Load dataset
        df = pd.read_csv('csv/student-scores.csv')
        print("Dataset loaded successfully.")
        print(f"Shape: {df.shape}")
        print(df.head())

        # Drop irrelevant columns
        df = df.drop(columns=['id', 'first_name', 'last_name', 'email'])

        # Handle boolean columns
        for col in ['part_time_job', 'extracurricular_activities']:
            df[col] = df[col].astype(int)

        # Calculate total score
        subject_cols = ['math_score', 'history_score', 'physics_score',
                        'chemistry_score', 'biology_score', 'english_score', 'geography_score']
        df['total_score'] = df[subject_cols].sum(axis=1)

        # Derive grade
        def assign_grade(score):
            if score >= 650:
                return 'A'
            elif score >= 600:
                return 'B'
            elif score >= 550:
                return 'C'
            elif score >= 500:
                return 'D'
            else:
                return 'F'
        df['grade'] = df['total_score'].apply(assign_grade)

        # Generate synthetic career aspirations
        np.random.seed(42)
        df['career_aspiration'] = np.random.choice(
            ['Engineer', 'Doctor', 'Artist', 'Teacher', 'Scientist'],
            size=len(df)
        )

        # One-Hot Encode gender
        df = pd.get_dummies(df, columns=['gender'], drop_first=True)

        # Label Encode grade and career aspiration
        grade_encoder = LabelEncoder()
        df['grade_encoded'] = grade_encoder.fit_transform(df['grade'])

        aspiration_encoder = LabelEncoder()
        df['career_aspiration_encoded'] = aspiration_encoder.fit_transform(df['career_aspiration'])

        # Define features and targets
        features = ['part_time_job', 'absence_days', 'math_score', 'history_score',
                    'physics_score', 'chemistry_score', 'biology_score', 'english_score',
                    'geography_score', 'gender_male']
        X = df[features]
        y_score = df['total_score']
        y_grade = df['grade_encoded']
        y_aspiration = df['career_aspiration_encoded']

        # Save feature names
        os.makedirs('trained_data', exist_ok=True)
        joblib.dump(features, 'trained_data/model_features.pkl')

        # Split and train regression model for total score
        X_train, X_test, y_train, y_test = train_test_split(X, y_score, test_size=0.2, random_state=42)
        reg_model = RandomForestRegressor(random_state=42)
        reg_model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, reg_model.predict(X_test)))
        print(f"\nTotal Score Model RMSE: {rmse:.2f}")
        joblib.dump(reg_model, 'trained_data/total_score_model.pkl')

        # Train classification model for grade
        X_train, X_test, y_train, y_test = train_test_split(X, y_grade, test_size=0.2, random_state=42, stratify=y_grade)
        grade_model = RandomForestClassifier(random_state=42)
        grade_model.fit(X_train, y_train)
        y_pred = grade_model.predict(X_test)
        print(f"\nGrade Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(classification_report(y_test, y_pred, target_names=grade_encoder.classes_))
        joblib.dump(grade_model, 'trained_data/grade_model.pkl')
        joblib.dump(grade_encoder, 'trained_data/grade_label_encoder.pkl')

        # Train classification model for career aspiration
        X_train, X_test, y_train, y_test = train_test_split(X, y_aspiration, test_size=0.2, random_state=42, stratify=y_aspiration)
        asp_model = RandomForestClassifier(random_state=42)
        asp_model.fit(X_train, y_train)
        y_pred = asp_model.predict(X_test)
        print(f"\nCareer Aspiration Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(classification_report(y_test, y_pred, target_names=aspiration_encoder.classes_))
        joblib.dump(asp_model, 'trained_data/career_aspiration_model.pkl')
        joblib.dump(aspiration_encoder, 'trained_data/career_aspiration_label_encoder.pkl')

        print("\n✅ All models trained and saved successfully.")

    except FileNotFoundError:
        print("❌ 'csv/student-scores.csv' not found. Please check the path.")
    except Exception as e:
        print(f"❌ Error during training: {e}")

if __name__ == "__main__":
    train_model()
