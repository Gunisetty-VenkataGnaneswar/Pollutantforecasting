import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json

class PollutionForecaster:
    def __init__(self):
        """Initialize the pollution forecasting system with GenAI capabilities."""
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'hour', 'day_of_week', 'month', 'temperature', 'humidity', 
            'wind_speed', 'pressure', 'traffic_intensity', 'industrial_activity',
            'previous_pm25', 'previous_pm10', 'previous_co', 'previous_no2', 'previous_o3'
        ]
        self.target_columns = ['pm25', 'pm10', 'co', 'no2', 'o3', 'aqi']
        
        # Policy recommendation database
        self.policy_database = self._load_policy_database()
        
        # Initialize models for each pollutant
        for pollutant in self.target_columns:
            self.models[pollutant] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.scalers[pollutant] = StandardScaler()
        
        # Historical data for training
        self.historical_data = []
        self.is_trained = False
        
    def _load_policy_database(self):
        """Load policy recommendation database based on pollution levels."""
        return {
            'emergency_policies': {
                'aqi_200_plus': [
                    "Implement emergency vehicle restrictions (odd-even system)",
                    "Close schools and public institutions",
                    "Activate emergency air quality protocols",
                    "Deploy mobile air purification units",
                    "Issue public health warnings"
                ],
                'aqi_150_200': [
                    "Implement traffic congestion pricing",
                    "Increase public transportation frequency",
                    "Restrict heavy vehicle movement",
                    "Activate industrial emission controls",
                    "Issue sensitive group warnings"
                ]
            },
            'short_term_policies': {
                'traffic_reduction': [
                    "Implement carpooling incentives",
                    "Increase parking fees in city centers",
                    "Promote remote work policies",
                    "Enhance public transportation",
                    "Implement bike-sharing programs"
                ],
                'industrial_control': [
                    "Enforce stricter emission standards",
                    "Implement real-time monitoring systems",
                    "Require pollution control equipment upgrades",
                    "Establish emission trading schemes",
                    "Conduct regular compliance audits"
                ],
                'urban_planning': [
                    "Increase green spaces and urban forests",
                    "Implement green building standards",
                    "Create low-emission zones",
                    "Develop pedestrian-friendly areas",
                    "Install air purification systems"
                ]
            },
            'long_term_policies': {
                'infrastructure': [
                    "Invest in renewable energy infrastructure",
                    "Develop smart city technologies",
                    "Create comprehensive air quality monitoring networks",
                    "Build sustainable transportation systems",
                    "Implement circular economy practices"
                ],
                'policy_framework': [
                    "Establish air quality standards and regulations",
                    "Create environmental impact assessment requirements",
                    "Develop climate action plans",
                    "Implement carbon pricing mechanisms",
                    "Establish international cooperation agreements"
                ]
            }
        }
    
    def generate_synthetic_data(self, days=365):
        """Generate synthetic training data with realistic patterns."""
        data = []
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days * 24):  # Hourly data
            current_time = base_date + timedelta(hours=i)
            
            # Base environmental factors
            hour = current_time.hour
            day_of_week = current_time.weekday()
            month = current_time.month
            
            # Simulate weather conditions
            temperature = 15 + 10 * np.sin(2 * np.pi * (current_time.timetuple().tm_yday - 172) / 365) + random.uniform(-5, 5)
            humidity = 60 + 20 * np.sin(2 * np.pi * hour / 24) + random.uniform(-10, 10)
            wind_speed = 5 + 3 * random.random()
            pressure = 1013 + random.uniform(-20, 20)
            
            # Traffic patterns
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                traffic_intensity = random.uniform(0.7, 1.0)
            elif 22 <= hour or hour <= 5:
                traffic_intensity = random.uniform(0.1, 0.3)
            else:
                traffic_intensity = random.uniform(0.4, 0.6)
            
            # Industrial activity (lower on weekends)
            if day_of_week < 5:
                industrial_activity = random.uniform(0.6, 1.0)
            else:
                industrial_activity = random.uniform(0.2, 0.5)
            
            # Previous values (simulated)
            previous_pm25 = 15 + 5 * random.random()
            previous_pm10 = 30 + 10 * random.random()
            previous_co = 1.0 + 0.5 * random.random()
            previous_no2 = 40 + 15 * random.random()
            previous_o3 = 50 + 20 * random.random()
            
            # Calculate current pollutant levels
            pm25 = self._calculate_pollutant_level('pm25', hour, day_of_week, month, 
                                                 temperature, humidity, wind_speed, pressure,
                                                 traffic_intensity, industrial_activity,
                                                 previous_pm25, previous_pm10, previous_co, previous_no2, previous_o3)
            
            pm10 = self._calculate_pollutant_level('pm10', hour, day_of_week, month,
                                                 temperature, humidity, wind_speed, pressure,
                                                 traffic_intensity, industrial_activity,
                                                 previous_pm25, previous_pm10, previous_co, previous_no2, previous_o3)
            
            co = self._calculate_pollutant_level('co', hour, day_of_week, month,
                                               temperature, humidity, wind_speed, pressure,
                                               traffic_intensity, industrial_activity,
                                               previous_pm25, previous_pm10, previous_co, previous_no2, previous_o3)
            
            no2 = self._calculate_pollutant_level('no2', hour, day_of_week, month,
                                                temperature, humidity, wind_speed, pressure,
                                                traffic_intensity, industrial_activity,
                                                previous_pm25, previous_pm10, previous_co, previous_no2, previous_o3)
            
            o3 = self._calculate_pollutant_level('o3', hour, day_of_week, month,
                                               temperature, humidity, wind_speed, pressure,
                                               traffic_intensity, industrial_activity,
                                               previous_pm25, previous_pm10, previous_co, previous_no2, previous_o3)
            
            # Calculate AQI
            aqi = self._calculate_aqi(pm25, pm10, co, no2, o3)
            
            data.append({
                'timestamp': current_time,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'pressure': pressure,
                'traffic_intensity': traffic_intensity,
                'industrial_activity': industrial_activity,
                'previous_pm25': previous_pm25,
                'previous_pm10': previous_pm10,
                'previous_co': previous_co,
                'previous_no2': previous_no2,
                'previous_o3': previous_o3,
                'pm25': pm25,
                'pm10': pm10,
                'co': co,
                'no2': no2,
                'o3': o3,
                'aqi': aqi
            })
        
        return pd.DataFrame(data)
    
    def _calculate_pollutant_level(self, pollutant, hour, day_of_week, month,
                                 temperature, humidity, wind_speed, pressure,
                                 traffic_intensity, industrial_activity,
                                 previous_pm25, previous_pm10, previous_co, previous_no2, previous_o3):
        """Calculate pollutant levels based on environmental factors."""
        base_levels = {
            'pm25': 15,
            'pm10': 30,
            'co': 1.0,
            'no2': 40,
            'o3': 50
        }
        
        # Time-based factors
        time_factor = 1.0
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            time_factor = 1.3
        elif 22 <= hour or hour <= 5:  # Night
            time_factor = 0.7
        
        # Weather factors
        weather_factor = 1.0
        if wind_speed > 10:  # High wind disperses pollutants
            weather_factor = 0.8
        if humidity > 80:  # High humidity can trap pollutants
            weather_factor = 1.2
        
        # Traffic and industrial factors
        traffic_factor = 1.0 + 0.5 * traffic_intensity
        industrial_factor = 1.0 + 0.3 * industrial_activity
        
        # Seasonal factors
        seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * (month - 6) / 12)
        
        # Calculate final level
        base = base_levels[pollutant]
        level = base * time_factor * weather_factor * traffic_factor * industrial_factor * seasonal_factor
        
        # Add some randomness
        level *= random.uniform(0.8, 1.2)
        
        return max(0, level)
    
    def _calculate_aqi(self, pm25, pm10, co, no2, o3):
        """Calculate Air Quality Index."""
        # Simplified AQI calculation
        aqi_values = []
        
        # PM2.5 AQI
        if pm25 <= 12.0:
            aqi_pm25 = ((50 - 0) / (12.0 - 0)) * (pm25 - 0) + 0
        elif pm25 <= 35.4:
            aqi_pm25 = ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
        elif pm25 <= 55.4:
            aqi_pm25 = ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
        elif pm25 <= 150.4:
            aqi_pm25 = ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151
        elif pm25 <= 250.4:
            aqi_pm25 = ((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201
        else:
            aqi_pm25 = ((500 - 301) / (500.4 - 250.5)) * (pm25 - 250.5) + 301
        
        aqi_values.append(aqi_pm25)
        
        # PM10 AQI
        if pm10 <= 54:
            aqi_pm10 = ((50 - 0) / (54 - 0)) * (pm10 - 0) + 0
        elif pm10 <= 154:
            aqi_pm10 = ((100 - 51) / (154 - 55)) * (pm10 - 55) + 51
        elif pm10 <= 254:
            aqi_pm10 = ((150 - 101) / (254 - 155)) * (pm10 - 155) + 101
        elif pm10 <= 354:
            aqi_pm10 = ((200 - 151) / (354 - 255)) * (pm10 - 255) + 151
        elif pm10 <= 424:
            aqi_pm10 = ((300 - 201) / (424 - 355)) * (pm10 - 355) + 201
        else:
            aqi_pm10 = ((500 - 301) / (604 - 425)) * (pm10 - 425) + 301
        
        aqi_values.append(aqi_pm10)
        
        return max(aqi_values)
    
    def train_models(self):
        """Train the forecasting models with synthetic data."""
        print("Generating training data...")
        training_data = self.generate_synthetic_data(days=365)
        
        print("Training models...")
        for pollutant in self.target_columns:
            # Prepare features and target
            X = training_data[self.feature_columns]
            y = training_data[pollutant]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scalers[pollutant].fit_transform(X_train)
            X_test_scaled = self.scalers[pollutant].transform(X_test)
            
            # Train model
            self.models[pollutant].fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = self.models[pollutant].score(X_train_scaled, y_train)
            test_score = self.models[pollutant].score(X_test_scaled, y_test)
            
            print(f"{pollutant.upper()} - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        self.is_trained = True
        print("All models trained successfully!")
    
    def predict_next_hours(self, current_data, hours=6):
        """Predict air quality for the next N hours."""
        if not self.is_trained:
            self.train_models()
        
        predictions = []
        current_time = datetime.now()
        
        for hour in range(1, hours + 1):
            future_time = current_time + timedelta(hours=hour)
            
            # Prepare features for prediction
            features = {
                'hour': future_time.hour,
                'day_of_week': future_time.weekday(),
                'month': future_time.month,
                'temperature': 20 + 5 * np.sin(2 * np.pi * hour / 24) + random.uniform(-2, 2),
                'humidity': 60 + 10 * random.random(),
                'wind_speed': 5 + 2 * random.random(),
                'pressure': 1013 + random.uniform(-10, 10),
                'traffic_intensity': 0.5 + 0.3 * random.random(),
                'industrial_activity': 0.6 + 0.3 * random.random(),
                'previous_pm25': current_data['pm25'],
                'previous_pm10': current_data['pm10'],
                'previous_co': current_data['co'],
                'previous_no2': current_data['no2'],
                'previous_o3': current_data['o3']
            }
            
            # Make predictions for each pollutant
            prediction = {}
            for pollutant in self.target_columns:
                feature_vector = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
                feature_vector_scaled = self.scalers[pollutant].transform(feature_vector)
                prediction[pollutant] = max(0, self.models[pollutant].predict(feature_vector_scaled)[0])
            
            # Calculate AQI for the prediction
            prediction['aqi'] = self._calculate_aqi(
                prediction['pm25'], prediction['pm10'], 
                prediction['co'], prediction['no2'], prediction['o3']
            )
            
            prediction['timestamp'] = future_time
            predictions.append(prediction)
        
        return predictions
    
    def generate_policy_recommendations(self, current_data, forecast_data):
        """Generate AI-powered policy recommendations based on current and forecasted data."""
        recommendations = {
            'emergency_actions': [],
            'short_term_policies': [],
            'long_term_policies': [],
            'priority_level': 'low',
            'confidence_score': 0.0,
            'economic_impact': 'minimal',
            'implementation_timeline': '1-3 months'
        }
        
        # Analyze current AQI
        current_aqi = current_data['aqi']
        
        # Analyze forecast trends
        forecast_aqis = [f['aqi'] for f in forecast_data]
        max_forecast_aqi = max(forecast_aqis)
        avg_forecast_aqi = np.mean(forecast_aqis)
        trend = 'increasing' if forecast_aqis[-1] > forecast_aqis[0] else 'decreasing'
        
        # Determine priority level
        if max_forecast_aqi >= 200 or current_aqi >= 200:
            recommendations['priority_level'] = 'critical'
            recommendations['emergency_actions'] = self.policy_database['emergency_policies']['aqi_200_plus']
            recommendations['confidence_score'] = 0.95
            recommendations['economic_impact'] = 'high'
            recommendations['implementation_timeline'] = 'immediate'
        elif max_forecast_aqi >= 150 or current_aqi >= 150:
            recommendations['priority_level'] = 'high'
            recommendations['emergency_actions'] = self.policy_database['emergency_policies']['aqi_150_200']
            recommendations['confidence_score'] = 0.85
            recommendations['economic_impact'] = 'moderate'
            recommendations['implementation_timeline'] = '1-2 weeks'
        elif max_forecast_aqi >= 100 or current_aqi >= 100:
            recommendations['priority_level'] = 'medium'
            recommendations['confidence_score'] = 0.75
            recommendations['economic_impact'] = 'moderate'
            recommendations['implementation_timeline'] = '1-3 months'
        else:
            recommendations['priority_level'] = 'low'
            recommendations['confidence_score'] = 0.60
            recommendations['economic_impact'] = 'minimal'
            recommendations['implementation_timeline'] = '3-6 months'
        
        # Generate specific policy recommendations
        if current_data['traffic'] == 'high' or current_data['traffic'] == 'very_high':
            recommendations['short_term_policies'].extend(
                self.policy_database['short_term_policies']['traffic_reduction']
            )
        
        # Add industrial control policies if needed
        if current_data.get('industrial_activity', 0) > 0.7:
            recommendations['short_term_policies'].extend(
                self.policy_database['short_term_policies']['industrial_control']
            )
        
        # Add urban planning policies
        recommendations['short_term_policies'].extend(
            self.policy_database['short_term_policies']['urban_planning']
        )
        
        # Add long-term infrastructure policies
        recommendations['long_term_policies'].extend(
            self.policy_database['long_term_policies']['infrastructure']
        )
        
        # Add policy framework recommendations
        recommendations['long_term_policies'].extend(
            self.policy_database['long_term_policies']['policy_framework']
        )
        
        # Add AI-generated insights
        recommendations['ai_insights'] = self._generate_ai_insights(current_data, forecast_data, trend)
        
        return recommendations
    
    def _generate_ai_insights(self, current_data, forecast_data, trend):
        """Generate AI-powered insights about the air quality situation."""
        insights = []
        
        # Trend analysis
        if trend == 'increasing':
            insights.append("⚠️ Air quality is projected to deteriorate over the next 6 hours")
        else:
            insights.append("✅ Air quality is projected to improve over the next 6 hours")
        
        # Pollutant analysis
        pollutants = ['pm25', 'pm10', 'co', 'no2', 'o3']
        max_pollutant = max(pollutants, key=lambda x: current_data[x])
        insights.append(f"🔍 Primary pollutant of concern: {max_pollutant.upper()}")
        
        # Time-based insights
        current_hour = datetime.now().hour
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
            insights.append("🚗 Rush hour traffic is likely contributing to elevated pollution levels")
        
        # Weather insights
        if current_data.get('weather') == 'smog':
            insights.append("🌫️ Smog conditions are significantly impacting air quality")
        elif current_data.get('weather') == 'rainy':
            insights.append("🌧️ Rain is helping to clean the air")
        
        # Health impact insights
        aqi = current_data['aqi']
        if aqi > 150:
            insights.append("🏥 Sensitive groups should avoid outdoor activities")
        elif aqi > 100:
            insights.append("⚠️ Consider limiting outdoor activities for sensitive individuals")
        
        return insights
    
    def save_models(self, filepath='models/'):
        """Save trained models to disk."""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        for pollutant, model in self.models.items():
            model_path = os.path.join(filepath, f'{pollutant}_model.pkl')
            scaler_path = os.path.join(filepath, f'{pollutant}_scaler.pkl')
            
            joblib.dump(model, model_path)
            joblib.dump(self.scalers[pollutant], scaler_path)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath='models/'):
        """Load trained models from disk."""
        import os
        
        for pollutant in self.target_columns:
            model_path = os.path.join(filepath, f'{pollutant}_model.pkl')
            scaler_path = os.path.join(filepath, f'{pollutant}_scaler.pkl')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[pollutant] = joblib.load(model_path)
                self.scalers[pollutant] = joblib.load(scaler_path)
        
        self.is_trained = True
        print("Models loaded successfully!")
