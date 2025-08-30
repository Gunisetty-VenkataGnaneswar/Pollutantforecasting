import cv2
import numpy as np
from PIL import Image
import streamlit as st
from datetime import datetime
import os

class ImagePollutionAnalyzer:
    def __init__(self):
        """Initialize the image pollution analyzer."""
        self.camera = None
        self.analysis_results = {}
        
    def start_camera(self):
        """Start the camera for image capture."""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                st.error("❌ Could not open camera. Please check if camera is connected.")
                return False
            return True
        except Exception as e:
            st.error(f"❌ Camera error: {e}")
            return False
    
    def capture_image(self):
        """Capture an image from the camera."""
        if self.camera is None:
            st.error("❌ Camera not initialized. Please start camera first.")
            return None
        
        try:
            ret, frame = self.camera.read()
            if not ret:
                st.error("❌ Failed to capture image from camera.")
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        except Exception as e:
            st.error(f"❌ Image capture error: {e}")
            return None
    
    def analyze_visibility(self, image):
        """Analyze visibility and haze in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate visibility metrics
        # 1. Contrast analysis
        contrast = np.std(gray)
        
        # 2. Brightness analysis
        brightness = np.mean(gray)
        
        # 3. Edge detection for clarity
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 4. Color analysis for haze detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:, :, 1])
        
        # Determine visibility quality
        if contrast < 30 and saturation < 50:
            visibility = "Poor - Heavy Haze/Smog"
            pollution_level = "High"
        elif contrast < 50 and saturation < 80:
            visibility = "Moderate - Light Haze"
            pollution_level = "Moderate"
        else:
            visibility = "Good - Clear Visibility"
            pollution_level = "Low"
        
        return {
            'visibility': visibility,
            'pollution_level': pollution_level,
            'contrast': contrast,
            'brightness': brightness,
            'edge_density': edge_density,
            'saturation': saturation
        }
    
    def analyze_sky_color(self, image):
        """Analyze sky color for pollution indicators."""
        # Focus on upper portion of image (sky area)
        height, width = image.shape[:2]
        sky_region = image[0:int(height*0.4), :, :]
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(sky_region, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(sky_region, cv2.COLOR_RGB2LAB)
        
        # Analyze color characteristics
        blue_channel = sky_region[:, :, 2]  # Blue channel in RGB
        blue_mean = np.mean(blue_channel)
        
        # Analyze for brown/gray tint (pollution indicators)
        gray_region = cv2.cvtColor(sky_region, cv2.COLOR_RGB2GRAY)
        brown_mask = cv2.inRange(hsv, (10, 50, 50), (20, 255, 255))  # Brown/orange range
        gray_mask = cv2.inRange(gray_region, 100, 150)  # Gray range
        
        brown_ratio = np.sum(brown_mask > 0) / (brown_mask.shape[0] * brown_mask.shape[1])
        gray_ratio = np.sum(gray_mask > 0) / (gray_mask.shape[0] * gray_mask.shape[1])
        
        # Determine sky quality
        if brown_ratio > 0.1 or gray_ratio > 0.3:
            sky_quality = "Polluted - Brown/Gray Tint"
            air_quality = "Poor"
        elif blue_mean < 100:
            sky_quality = "Hazy - Reduced Blue"
            air_quality = "Moderate"
        else:
            sky_quality = "Clear - Natural Blue"
            air_quality = "Good"
        
        return {
            'sky_quality': sky_quality,
            'air_quality': air_quality,
            'blue_intensity': blue_mean,
            'brown_ratio': brown_ratio,
            'gray_ratio': gray_ratio
        }
    
    def detect_smoke_plumes(self, image):
        """Detect potential smoke plumes or industrial emissions."""
        # Convert to different color spaces for smoke detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Smoke typically appears as white/gray regions with specific characteristics
        # Look for regions with low saturation and medium brightness
        smoke_mask = cv2.inRange(hsv, (0, 0, 100), (180, 30, 200))
        
        # Find contours of potential smoke regions
        contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        smoke_detected = False
        smoke_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small regions
                smoke_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                smoke_regions.append({
                    'area': area,
                    'position': (x, y, w, h)
                })
        
        return {
            'smoke_detected': smoke_detected,
            'smoke_regions': smoke_regions,
            'smoke_coverage': np.sum(smoke_mask > 0) / (smoke_mask.shape[0] * smoke_mask.shape[1])
        }
    
    def analyze_overall_pollution(self, image):
        """Comprehensive pollution analysis from image."""
        # Perform all analyses
        visibility_analysis = self.analyze_visibility(image)
        sky_analysis = self.analyze_sky_color(image)
        smoke_analysis = self.detect_smoke_plumes(image)
        
        # Combine results for overall assessment
        pollution_indicators = []
        
        if visibility_analysis['pollution_level'] == 'High':
            pollution_indicators.append('Heavy Haze/Smog')
        elif visibility_analysis['pollution_level'] == 'Moderate':
            pollution_indicators.append('Light Haze')
        
        if sky_analysis['air_quality'] == 'Poor':
            pollution_indicators.append('Brown/Gray Sky')
        elif sky_analysis['air_quality'] == 'Moderate':
            pollution_indicators.append('Hazy Sky')
        
        if smoke_analysis['smoke_detected']:
            pollution_indicators.append('Smoke Plumes Detected')
        
        # Determine overall air quality
        if len(pollution_indicators) >= 2:
            overall_quality = "Poor"
            aqi_estimate = "150-200"
        elif len(pollution_indicators) == 1:
            overall_quality = "Moderate"
            aqi_estimate = "100-150"
        else:
            overall_quality = "Good"
            aqi_estimate = "50-100"
        
        return {
            'overall_quality': overall_quality,
            'aqi_estimate': aqi_estimate,
            'pollution_indicators': pollution_indicators,
            'visibility_analysis': visibility_analysis,
            'sky_analysis': sky_analysis,
            'smoke_analysis': smoke_analysis,
            'timestamp': datetime.now()
        }
    
    def save_image(self, image, filename=None):
        """Save captured image to disk."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pollution_analysis_{timestamp}.jpg"
        
        # Create images directory if it doesn't exist
        os.makedirs("images", exist_ok=True)
        filepath = os.path.join("images", filename)
        
        # Save image
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return filepath
    
    def release_camera(self):
        """Release camera resources."""
        if self.camera:
            self.camera.release()
            self.camera = None
