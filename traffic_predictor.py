import tensorflow as tf
import numpy as np
import pandas as pd
import requests
import json
import datetime
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# API Configuration
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "YOUR_GOOGLE_MAPS_API_KEY")

class TrafficAccidentPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self, num_samples=5000):
        """Generate synthetic data for training the model"""
        print("Generating synthetic training data...")
        
        # Generate random features
        np.random.seed(42)
        traffic_density = np.random.uniform(0, 1, num_samples)  # 0 to 1 scale
        avg_speed = np.random.normal(55, 15, num_samples)  # mph
        rush_hour = np.random.randint(0, 2, num_samples)  # binary: 0 or 1
        bad_weather = np.random.uniform(0, 1, num_samples)  # 0 to 1 scale
        road_curvature = np.random.uniform(0, 1, num_samples)  # 0 to 1 scale
        road_works = np.random.uniform(0, 0.3, num_samples)  # 0 to 0.3 scale
        day_of_week = np.random.randint(0, 7, num_samples)  # 0-6 (Monday to Sunday)
        hour_of_day = np.random.randint(0, 24, num_samples)  # 0-23
        
        # Create feature matrix
        X = np.column_stack([
            traffic_density, 
            avg_speed, 
            rush_hour, 
            bad_weather, 
            road_curvature, 
            road_works,
            day_of_week,
            hour_of_day
        ])
        
        # Generate target variable (accident probability)
        # Higher values in these conditions increase accident probability
        accident_prob = (
            0.3 * traffic_density + 
            0.1 * np.abs(avg_speed - 55) / 55 +  # Deviation from safe speed
            0.2 * rush_hour + 
            0.3 * bad_weather + 
            0.1 * road_curvature +
            0.2 * road_works +
            0.05 * (day_of_week == 5) +  # Slight increase on Fridays
            0.05 * (day_of_week == 6) +  # Slight increase on Saturdays
            0.1 * ((hour_of_day >= 16) & (hour_of_day <= 19))  # Increase during evening rush
        )
        
        # Add some randomness
        accident_prob += np.random.normal(0, 0.1, num_samples)
        
        # Clip between 0 and 1
        accident_prob = np.clip(accident_prob, 0, 1)
        
        # Convert to binary target (accident or no accident)
        # Using a threshold of 0.5 for demonstration
        y = (accident_prob > 0.5).astype(int)
        
        # Create pandas DataFrame for easier handling
        columns = [
            'traffic_density', 'avg_speed', 'rush_hour', 'bad_weather',
            'road_curvature', 'road_works', 'day_of_week', 'hour_of_day'
        ]
        df = pd.DataFrame(X, columns=columns)
        df['accident'] = y
        df['accident_prob'] = accident_prob  # Store the original probability too
        
        print(f"Generated {num_samples} samples with {sum(y)} accidents ({sum(y)/num_samples*100:.1f}%)")
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        X = df.drop(['accident', 'accident_prob'], axis=1, errors='ignore').values
        y = df['accident'].values
        
        # Normalize features
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """Build the TensorFlow model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def train(self, df=None, epochs=20):
        """Train the model with either provided or synthetic data"""
        if df is None:
            df = self.generate_synthetic_data()
        
        X_train, X_test, y_train, y_test = self.preprocess_data(df)
        
        # Build the model
        self.model = self.build_model(X_train.shape[1])
        
        # Train the model
        print("Training the model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate the model
        loss, accuracy, auc = self.model.evaluate(X_test, y_test)
        print(f"Test accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        return history
    
    def save_model(self, path="traffic_accident_model"):
        """Save the trained model"""
        if self.model is None:
            print("No model to save. Please train the model first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save the model architecture and weights
        self.model.save(path)
        
        # Save the scaler for future preprocessing
        with open(f"{path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path="traffic_accident_model"):
        """Load a trained model"""
        try:
            # Load the model
            self.model = tf.keras.models.load_model(path)
            
            # Load the scaler
            with open(f"{path}/scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # If we can't load the model, train a new one
            print("Training a new model...")
            self.train(epochs=10)
            self.save_model(path)
    
    def get_traffic_data(self, location):
        """
        Get or generate traffic data for a given location
        In a production environment, this would call real APIs
        """
        # Generate simulated data
        now = datetime.datetime.now()
        hour = now.hour
        weekday = now.weekday()
        
        # Simulate rush hour patterns
        is_rush_hour = 1 if (7 <= hour <= 9) or (16 <= hour <= 18) else 0
        
        # Simulate higher traffic on weekdays (0-4) vs weekends (5-6)
        traffic_factor = 0.8 if weekday < 5 else 0.5
        
        # Generate traffic density with some randomness
        traffic_base = 0.3 + (0.5 * traffic_factor * is_rush_hour)
        traffic_density = min(1.0, traffic_base + np.random.uniform(-0.1, 0.1))
        
        # Generate average speed (inversely related to traffic)
        avg_speed = 65 - (30 * traffic_density) + np.random.normal(0, 5)
        
        # Weather condition (simplified)
        bad_weather = np.random.uniform(0, 0.5)  # 0 to 0.5 scale for demonstration
        
        # Proximity to reported issues increases road_works value
        road_works = np.random.uniform(0, 0.3)
        
        # Road curvature (simulated)
        road_curvature = np.random.uniform(0, 1)
        
        return {
            "traffic_density": traffic_density,
            "avg_speed": avg_speed,
            "rush_hour": is_rush_hour,
            "bad_weather": bad_weather,
            "road_curvature": road_curvature,
            "road_works": road_works,
            "day_of_week": weekday,
            "hour_of_day": hour
        }
    
    def predict_accident(self, location):
        """Predict accident probability for a given location"""
        if self.model is None:
            print("No model available. Please train or load a model first.")
            return None
        
        # Get traffic data
        traffic_data = self.get_traffic_data(location)
        
        # Prepare the features for prediction
        features = np.array([[
            traffic_data["traffic_density"],
            traffic_data["avg_speed"],
            traffic_data["rush_hour"],
            traffic_data["bad_weather"],
            traffic_data["road_curvature"],
            traffic_data["road_works"],
            traffic_data["day_of_week"],
            traffic_data["hour_of_day"]
        ]])
        
        # Normalize the features using the same scaler used during training
        normalized_features = self.scaler.transform(features)
        
        # Make prediction
        prediction = float(self.model.predict(normalized_features)[0][0])
        
        # Add traffic data to the results
        traffic_data.update({
            "location": location,
            "accident_probability": prediction,
            "high_risk": prediction > 0.7,
            "medium_risk": 0.3 < prediction <= 0.7,
            "low_risk": prediction <= 0.3,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        return traffic_data


# Example use when running this file directly
if __name__ == "__main__":
    # Create the predictor
    predictor = TrafficAccidentPredictor()
    
    # Train the model with synthetic data
    predictor.train(epochs=10)
    
    # Save the model for future use
    predictor.save_model()
    
    # Test a prediction
    test_location = {"lat": 37.7749, "lng": -122.4194}
    result = predictor.predict_accident(test_location)
    
    print(f"Accident probability: {result['accident_probability']:.2f}")
    if result["high_risk"]:
        print("Risk level: HIGH")
    elif result["medium_risk"]:
        print("Risk level: MEDIUM")
    else:
        print("Risk level: LOW")