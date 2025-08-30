import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
import requests
import geocoder

class AirQualitySensor:
    def __init__(self):
        """Initialize the air quality sensor system with location detection."""
        self.sensor_locations = {
            'New York': {'lat': 40.7128, 'lon': -74.0060, 'elevation': 10, 'city_type': 'metropolitan'},
            'Los Angeles': {'lat': 34.0522, 'lon': -118.2437, 'elevation': 93, 'city_type': 'metropolitan'},
            'Chicago': {'lat': 41.8781, 'lon': -87.6298, 'elevation': 179, 'city_type': 'industrial'},
            'Houston': {'lat': 29.7604, 'lon': -95.3698, 'elevation': 13, 'city_type': 'industrial'},
            'Phoenix': {'lat': 33.4484, 'lon': -112.0740, 'elevation': 331, 'city_type': 'desert'},
            'Philadelphia': {'lat': 39.9526, 'lon': -75.1652, 'elevation': 12, 'city_type': 'metropolitan'},
            'San Antonio': {'lat': 29.4241, 'lon': -98.4936, 'elevation': 198, 'city_type': 'mixed'},
            'San Diego': {'lat': 32.7157, 'lon': -117.1611, 'elevation': 19, 'city_type': 'coastal'}
        }
        
        # Enhanced pollutant profiles for different city types
        self.city_pollutant_profiles = {
            'metropolitan': {
                'primary_sources': ['Vehicle emissions', 'Industrial activities', 'Construction'],
                'dominant_pollutants': ['NO2', 'PM2.5', 'CO'],
                'seasonal_factors': {'summer': 'High O3', 'winter': 'High PM2.5'},
                'traffic_impact': 'high',
                'industrial_impact': 'moderate'
            },
            'industrial': {
                'primary_sources': ['Industrial facilities', 'Power plants', 'Heavy machinery'],
                'dominant_pollutants': ['SO2', 'PM10', 'NO2'],
                'seasonal_factors': {'summer': 'High O3', 'winter': 'High PM10'},
                'traffic_impact': 'moderate',
                'industrial_impact': 'high'
            },
            'desert': {
                'primary_sources': ['Dust storms', 'Vehicle emissions', 'Agricultural burning'],
                'dominant_pollutants': ['PM10', 'PM2.5', 'O3'],
                'seasonal_factors': {'summer': 'High O3', 'spring': 'High PM10'},
                'traffic_impact': 'moderate',
                'industrial_impact': 'low'
            },
            'coastal': {
                'primary_sources': ['Marine traffic', 'Tourism', 'Local industry'],
                'dominant_pollutants': ['O3', 'PM2.5', 'NO2'],
                'seasonal_factors': {'summer': 'High O3', 'winter': 'Moderate PM2.5'},
                'traffic_impact': 'moderate',
                'industrial_impact': 'low'
            },
            'mixed': {
                'primary_sources': ['Mixed urban sources', 'Transportation', 'Industry'],
                'dominant_pollutants': ['PM2.5', 'NO2', 'O3'],
                'seasonal_factors': {'summer': 'High O3', 'winter': 'High PM2.5'},
                'traffic_impact': 'moderate',
                'industrial_impact': 'moderate'
            }
        }
        
        # Historical data patterns for realistic simulation
        self.historical_patterns = self._load_historical_patterns()
        self.current_location = self._detect_current_location()
        self.time_of_day = 0
        self.weather_conditions = 'clear'
        self.traffic_intensity = 'moderate'
        
    def _detect_current_location(self):
        """Detect current location using IP geolocation."""
        try:
            # Try to get location from IP
            g = geocoder.ip('me')
            if g.ok:
                detected_city = g.city
                detected_state = g.state
                detected_country = g.country
                
                print(f"📍 Detected location: {detected_city}, {detected_state}, {detected_country}")
                
                # Try to match with our supported cities
                for city in self.sensor_locations.keys():
                    if city.lower() in detected_city.lower():
                        print(f"✅ Matched with supported city: {city}")
                        return city
                
                # If no exact match, find the closest city
                closest_city = self._find_closest_city(g.lat, g.lng)
                print(f"🌍 Using closest supported city: {closest_city}")
                return closest_city
                
        except Exception as e:
            print(f"⚠️ Could not detect location automatically: {e}")
        
        # Default to New York if detection fails
        print("📍 Using default location: New York")
        return 'New York'
    
    def _find_closest_city(self, lat, lng):
        """Find the closest supported city to the detected coordinates."""
        min_distance = float('inf')
        closest_city = 'New York'
        
        for city, coords in self.sensor_locations.items():
            distance = self._calculate_distance(lat, lng, coords['lat'], coords['lon'])
            if distance < min_distance:
                min_distance = distance
                closest_city = city
        
        return closest_city
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates using Haversine formula."""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
        
    def _load_historical_patterns(self):
        """Load historical air quality patterns for different cities."""
        patterns = {}
        for city in self.sensor_locations.keys():
            city_type = self.sensor_locations[city]['city_type']
            profile = self.city_pollutant_profiles[city_type]
            
            # Adjust base levels based on city type
            if city_type == 'industrial':
                base_pm25 = random.uniform(15, 35)
                base_pm10 = random.uniform(25, 55)
                base_co = random.uniform(1.0, 3.0)
                base_no2 = random.uniform(30, 80)
                base_o3 = random.uniform(25, 70)
                base_so2 = random.uniform(5, 15)
            elif city_type == 'desert':
                base_pm25 = random.uniform(20, 40)
                base_pm10 = random.uniform(40, 80)
                base_co = random.uniform(0.5, 1.5)
                base_no2 = random.uniform(15, 45)
                base_o3 = random.uniform(40, 90)
                base_so2 = random.uniform(2, 8)
            elif city_type == 'coastal':
                base_pm25 = random.uniform(8, 20)
                base_pm10 = random.uniform(15, 35)
                base_co = random.uniform(0.3, 1.0)
                base_no2 = random.uniform(20, 50)
                base_o3 = random.uniform(35, 85)
                base_so2 = random.uniform(1, 5)
            else:  # metropolitan and mixed
                base_pm25 = random.uniform(10, 25)
                base_pm10 = random.uniform(20, 45)
                base_co = random.uniform(0.5, 2.0)
                base_no2 = random.uniform(25, 65)
                base_o3 = random.uniform(30, 80)
                base_so2 = random.uniform(3, 10)
            
            patterns[city] = {
                'base_pm25': base_pm25,
                'base_pm10': base_pm10,
                'base_co': base_co,
                'base_no2': base_no2,
                'base_o3': base_o3,
                'base_so2': base_so2,
                'traffic_factor': random.uniform(0.8, 1.5),
                'industrial_factor': random.uniform(0.7, 1.3),
                'weather_sensitivity': random.uniform(0.9, 1.2),
                'city_type': city_type,
                'pollutant_profile': profile
            }
        return patterns
    
    def set_location(self, location):
        """Set the monitoring location."""
        if location in self.sensor_locations:
            self.current_location = location
            return True
        return False
    
    def set_weather_conditions(self, conditions):
        """Set weather conditions affecting air quality."""
        self.weather_conditions = conditions
    
    def set_traffic_intensity(self, intensity):
        """Set traffic intensity level."""
        self.traffic_intensity = intensity
    
    def get_location_info(self):
        """Get detailed information about the current location."""
        location_data = self.sensor_locations[self.current_location]
        pattern = self.historical_patterns[self.current_location]
        
        return {
            'city': self.current_location,
            'coordinates': {
                'latitude': location_data['lat'],
                'longitude': location_data['lon'],
                'elevation': location_data['elevation']
            },
            'city_type': location_data['city_type'],
            'pollutant_profile': pattern['pollutant_profile'],
            'typical_pollutants': pattern['pollutant_profile']['dominant_pollutants'],
            'primary_sources': pattern['pollutant_profile']['primary_sources'],
            'seasonal_factors': pattern['pollutant_profile']['seasonal_factors']
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
        """Get current air quality data with realistic variations and location-specific pollutants."""
        pattern = self.historical_patterns[self.current_location]
        
        # Calculate environmental factors
        time_factor = self._calculate_time_factor()
        weather_factor = self._calculate_weather_factor()
        traffic_factor = self._calculate_traffic_factor()
        
        # Add random variations
        random_factor = random.uniform(0.8, 1.2)
        
        # Calculate pollutant concentrations based on city type
        city_type = pattern['city_type']
        
        # Base calculations
        pm25 = pattern['base_pm25'] * time_factor * weather_factor * traffic_factor * random_factor
        pm10 = pattern['base_pm10'] * time_factor * weather_factor * traffic_factor * random_factor
        co = pattern['base_co'] * time_factor * traffic_factor * random_factor
        no2 = pattern['base_no2'] * time_factor * traffic_factor * random_factor
        o3 = pattern['base_o3'] * weather_factor * random_factor
        so2 = pattern['base_so2'] * random_factor
        
        # City-specific adjustments
        if city_type == 'industrial':
            so2 *= 1.5  # Higher SO2 in industrial areas
            pm10 *= 1.3  # Higher PM10 from industrial activities
        elif city_type == 'desert':
            pm10 *= 1.8  # Much higher PM10 from dust
            pm25 *= 1.4  # Higher PM2.5 from dust storms
        elif city_type == 'coastal':
            o3 *= 1.2  # Higher O3 due to sea breeze
            so2 *= 0.7  # Lower SO2 in coastal areas
        
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
            'coordinates': self.sensor_locations[self.current_location],
            'city_type': city_type,
            'dominant_pollutants': location_info['typical_pollutants'],
            'primary_sources': location_info['primary_sources'],
            'seasonal_factors': location_info['seasonal_factors'],
            'pollution_sources': {
                'traffic': 'High' if traffic_factor > 1.2 else 'Moderate' if traffic_factor > 0.8 else 'Low',
                'industrial': 'High' if city_type == 'industrial' else 'Moderate' if city_type in ['metropolitan', 'mixed'] else 'Low',
                'natural': 'High' if city_type == 'desert' else 'Moderate' if city_type == 'coastal' else 'Low'
            }
        }
    
    def get_historical_data(self, days=30):
        """Get historical air quality data for analysis."""
        data = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            # Simulate historical data with realistic patterns
            pattern = self.historical_patterns[self.current_location]
            
            # Add seasonal and weekly patterns
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * date.weekday() / 7)
            
            pm25 = pattern['base_pm25'] * seasonal_factor * weekly_factor * random.uniform(0.7, 1.3)
            pm10 = pattern['base_pm10'] * seasonal_factor * weekly_factor * random.uniform(0.7, 1.3)
            co = pattern['base_co'] * weekly_factor * random.uniform(0.7, 1.3)
            no2 = pattern['base_no2'] * weekly_factor * random.uniform(0.7, 1.3)
            o3 = pattern['base_o3'] * seasonal_factor * random.uniform(0.7, 1.3)
            so2 = pattern['base_so2'] * random.uniform(0.7, 1.3)
            
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
            'locations': list(self.sensor_locations.keys()),
            'current_location': self.current_location,
            'location_info': self.get_location_info()
        }
