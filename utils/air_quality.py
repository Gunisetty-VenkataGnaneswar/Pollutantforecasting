import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
import requests
import geocoder

class AirQualitySensor:
    def __init__(self):
        """Initialize the air quality sensor system with dynamic location handling."""
        # Generic location template
        self.current_location = "Current Location"
        self.current_coordinates = {'lat': 0, 'lon': 0, 'elevation': 0}
        self.city_type = 'urban'
        
        # Generic pollutant profiles
        self.pollutant_profiles = {
            'urban': {
                'primary_sources': ['Vehicle emissions', 'Industrial activities', 'Construction'],
                'dominant_pollutants': ['NO2', 'PM2.5', 'CO'],
                'seasonal_factors': {'summer': 'High O3', 'winter': 'High PM2.5'},
                'base_levels': {
                    'pm25': random.uniform(10, 25),
                    'pm10': random.uniform(20, 45),
                    'co': random.uniform(0.5, 2.0),
                    'no2': random.uniform(25, 65),
                    'o3': random.uniform(30, 80),
                    'so2': random.uniform(3, 10)
                }
            },
            'industrial': {
                'primary_sources': ['Industrial facilities', 'Power plants', 'Heavy machinery'],
                'dominant_pollutants': ['SO2', 'PM10', 'NO2'],
                'seasonal_factors': {'summer': 'High O3', 'winter': 'High PM10'},
                'base_levels': {
                    'pm25': random.uniform(15, 35),
                    'pm10': random.uniform(25, 55),
                    'co': random.uniform(1.0, 3.0),
                    'no2': random.uniform(30, 80),
                    'o3': random.uniform(25, 70),
                    'so2': random.uniform(5, 15)
                }
            },
            'rural': {
                'primary_sources': ['Agricultural activities', 'Natural sources', 'Local traffic'],
                'dominant_pollutants': ['PM2.5', 'O3', 'PM10'],
                'seasonal_factors': {'summer': 'High O3', 'spring': 'High PM10'},
                'base_levels': {
                    'pm25': random.uniform(5, 15),
                    'pm10': random.uniform(10, 30),
                    'co': random.uniform(0.2, 0.8),
                    'no2': random.uniform(10, 30),
                    'o3': random.uniform(20, 60),
                    'so2': random.uniform(1, 5)
                }
            }
        }
        
        self.time_of_day = 0
        self.weather_conditions = 'clear'
        self.traffic_intensity = 'moderate'
        
        # Try to detect current location
        self._detect_current_location()
        
    def _detect_current_location(self):
        """Detect current location using IP geolocation."""
        try:
            # Try to get location from IP
            g = geocoder.ip('me')
            if g.ok:
                self.current_location = f"{g.city}, {g.state}, {g.country}"
                self.current_coordinates = {'lat': g.lat, 'lon': g.lng, 'elevation': 0}
                
                # Determine city type based on location characteristics
                if g.city and any(keyword in g.city.lower() for keyword in ['industrial', 'factory', 'plant']):
                    self.city_type = 'industrial'
                elif g.city and any(keyword in g.city.lower() for keyword in ['rural', 'farm', 'village']):
                    self.city_type = 'rural'
                else:
                    self.city_type = 'urban'
                
                print(f"📍 Detected location: {self.current_location}")
                print(f"🏙️ City type: {self.city_type}")
                
        except Exception as e:
            print(f"⚠️ Could not detect location automatically: {e}")
            print("📍 Using default location settings")
    
    def set_location(self, location_name=None):
        """Set the monitoring location."""
        if location_name:
            self.current_location = location_name
            # You can add logic here to determine city type based on location name
            if any(keyword in location_name.lower() for keyword in ['industrial', 'factory', 'plant']):
                self.city_type = 'industrial'
            elif any(keyword in location_name.lower() for keyword in ['rural', 'farm', 'village']):
                self.city_type = 'rural'
            else:
                self.city_type = 'urban'
        return True
    
    def set_weather_conditions(self, conditions):
        """Set weather conditions affecting air quality."""
        self.weather_conditions = conditions
    
    def set_traffic_intensity(self, intensity):
        """Set traffic intensity level."""
        self.traffic_intensity = intensity
    
    def get_location_info(self):
        """Get detailed information about the current location."""
        profile = self.pollutant_profiles[self.city_type]
        
        return {
            'city': self.current_location,
            'coordinates': self.current_coordinates,
            'city_type': self.city_type,
            'pollutant_profile': profile,
            'typical_pollutants': profile['dominant_pollutants'],
            'primary_sources': profile['primary_sources'],
            'seasonal_factors': profile['seasonal_factors']
        }
    
    def _calculate_time_factor(self):
        """Calculate time-based factors affecting air quality."""
        current_hour = datetime.now().hour
        
        # Rush hour effects
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
            time_factor = 1.3  # Higher pollution during rush hours
        elif 22 <= current_hour or current_hour <= 5:
            time_factor = 0.7  # Lower pollution during night
        else:
            time_factor = 1.0
        
        return time_factor
    
    def _calculate_weather_factor(self):
        """Calculate weather-based factors affecting air quality."""
        weather_factors = {
            'clear': 1.0,
            'cloudy': 0.9,
            'rainy': 0.7,  # Rain cleans the air
            'windy': 0.8,  # Wind disperses pollutants
            'foggy': 1.4,  # Fog traps pollutants
            'smog': 2.0    # Smog significantly increases pollution
        }
        return weather_factors.get(self.weather_conditions, 1.0)
    
    def _calculate_traffic_factor(self):
        """Calculate traffic-based factors affecting air quality."""
        traffic_factors = {
            'low': 0.6,
            'moderate': 1.0,
            'high': 1.4,
            'very_high': 1.8
        }
        return traffic_factors.get(self.traffic_intensity, 1.0)
    
    def _calculate_aqi(self, pm25, pm10, co, no2, o3, so2=None):
        """Calculate Air Quality Index based on pollutant concentrations."""
        # Enhanced AQI calculation including SO2
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
        
        # SO2 AQI (if available)
        if so2 is not None:
            if so2 <= 35:
                aqi_so2 = ((50 - 0) / (35 - 0)) * (so2 - 0) + 0
            elif so2 <= 75:
                aqi_so2 = ((100 - 51) / (75 - 36)) * (so2 - 36) + 51
            elif so2 <= 185:
                aqi_so2 = ((150 - 101) / (185 - 76)) * (so2 - 76) + 101
            elif so2 <= 304:
                aqi_so2 = ((200 - 151) / (304 - 186)) * (so2 - 186) + 151
            elif so2 <= 604:
                aqi_so2 = ((300 - 201) / (604 - 305)) * (so2 - 305) + 201
            else:
                aqi_so2 = ((500 - 301) / (1004 - 605)) * (so2 - 605) + 301
            
            aqi_values.append(aqi_so2)
        
        # Return the highest AQI value
        return max(aqi_values)
    
    def get_air_quality(self):
        """Get current air quality data with realistic variations."""
        profile = self.pollutant_profiles[self.city_type]
        base_levels = profile['base_levels']
        
        # Calculate environmental factors
        time_factor = self._calculate_time_factor()
        weather_factor = self._calculate_weather_factor()
        traffic_factor = self._calculate_traffic_factor()
        
        # Add random variations
        random_factor = random.uniform(0.8, 1.2)
        
        # Calculate pollutant concentrations
        pm25 = base_levels['pm25'] * time_factor * weather_factor * traffic_factor * random_factor
        pm10 = base_levels['pm10'] * time_factor * weather_factor * traffic_factor * random_factor
        co = base_levels['co'] * time_factor * traffic_factor * random_factor
        no2 = base_levels['no2'] * time_factor * traffic_factor * random_factor
        o3 = base_levels['o3'] * weather_factor * random_factor
        so2 = base_levels['so2'] * random_factor
        
        # City-specific adjustments
        if self.city_type == 'industrial':
            so2 *= 1.5  # Higher SO2 in industrial areas
            pm10 *= 1.3  # Higher PM10 from industrial activities
        elif self.city_type == 'rural':
            pm25 *= 0.8  # Lower PM2.5 in rural areas
            no2 *= 0.7   # Lower NO2 in rural areas
        
        # Calculate AQI
        aqi = self._calculate_aqi(pm25, pm10, co, no2, o3, so2)
        
        # Determine air quality status
        if aqi <= 50:
            status = "Good"
        elif aqi <= 100:
            status = "Moderate"
        elif aqi <= 150:
            status = "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            status = "Unhealthy"
        elif aqi <= 300:
            status = "Very Unhealthy"
        else:
            status = "Hazardous"
        
        # Get location-specific pollutant information
        location_info = self.get_location_info()
        
        return {
            'timestamp': datetime.now(),
            'location': self.current_location,
            'pm25': round(pm25, 2),
            'pm10': round(pm10, 2),
            'co': round(co, 2),
            'no2': round(no2, 2),
            'o3': round(o3, 2),
            'so2': round(so2, 2),
            'aqi': round(aqi, 1),
            'status': status,
            'weather': self.weather_conditions,
            'traffic': self.traffic_intensity,
            'coordinates': self.current_coordinates,
            'city_type': self.city_type,
            'dominant_pollutants': location_info['typical_pollutants'],
            'primary_sources': location_info['primary_sources'],
            'seasonal_factors': location_info['seasonal_factors'],
            'pollution_sources': {
                'traffic': 'High' if traffic_factor > 1.2 else 'Moderate' if traffic_factor > 0.8 else 'Low',
                'industrial': 'High' if self.city_type == 'industrial' else 'Moderate' if self.city_type == 'urban' else 'Low',
                'natural': 'High' if self.city_type == 'rural' else 'Moderate' if self.city_type == 'urban' else 'Low'
            }
        }
    
    def get_historical_data(self, days=30):
        """Get historical air quality data for analysis."""
        data = []
        profile = self.pollutant_profiles[self.city_type]
        base_levels = profile['base_levels']
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            
            # Add seasonal and weekly patterns
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * date.weekday() / 7)
            
            pm25 = base_levels['pm25'] * seasonal_factor * weekly_factor * random.uniform(0.7, 1.3)
            pm10 = base_levels['pm10'] * seasonal_factor * weekly_factor * random.uniform(0.7, 1.3)
            co = base_levels['co'] * weekly_factor * random.uniform(0.7, 1.3)
            no2 = base_levels['no2'] * weekly_factor * random.uniform(0.7, 1.3)
            o3 = base_levels['o3'] * seasonal_factor * random.uniform(0.7, 1.3)
            so2 = base_levels['so2'] * random.uniform(0.7, 1.3)
            
            aqi = self._calculate_aqi(pm25, pm10, co, no2, o3, so2)
            
            data.append({
                'date': date,
                'pm25': round(pm25, 2),
                'pm10': round(pm10, 2),
                'co': round(co, 2),
                'no2': round(no2, 2),
                'o3': round(o3, 2),
                'so2': round(so2, 2),
                'aqi': round(aqi, 1)
            })
        
        return data
    
    def get_sensor_metadata(self):
        """Get sensor system metadata."""
        return {
            'sensor_type': 'Multi-pollutant Environmental Sensor',
            'manufacturer': 'UrbanAirTech',
            'model': 'EAQ-2024',
            'calibration_date': datetime.now() - timedelta(days=30),
            'next_calibration': datetime.now() + timedelta(days=335),
            'accuracy': {
                'pm25': '±2 μg/m³',
                'pm10': '±5 μg/m³',
                'co': '±0.1 ppm',
                'no2': '±5 ppb',
                'o3': '±3 ppb',
                'so2': '±2 ppb'
            },
            'current_location': self.current_location,
            'city_type': self.city_type,
            'location_info': self.get_location_info()
        }
