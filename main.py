import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import csv
import time
import uuid
import random
from utils.air_quality import AirQualitySensor
from utils.forecasting import PollutionForecaster

class AirPollutionForecaster:
    def __init__(self):
        self.sensor = AirQualitySensor()
        self.forecaster = PollutionForecaster()
        self.is_monitoring = False
        self.session_id = None
        self.monitoring_duration = 24 * 60 * 60  # 24 hours in seconds
        self.start_time = None
        self.alert_count = 0
        self.last_alert_time = 0
        self.selected_location = None
        
        # Enhanced CSS styling for air quality monitoring
        st.markdown("""
            <style>
            .good-air {
                background-color: #d4edda !important;
                padding: 15px;
                border-radius: 8px;
                border: 3px solid #28a745;
                color: #155724;
                font-weight: bold;
                margin: 10px 0;
                text-align: center;
            }
            .moderate-air {
                background-color: #fff3cd !important;
                padding: 15px;
                border-radius: 8px;
                border: 3px solid #ffc107;
                color: #856404;
                font-weight: bold;
                margin: 10px 0;
                text-align: center;
            }
            .poor-air {
                background-color: #f8d7da !important;
                padding: 15px;
                border-radius: 8px;
                border: 3px solid #dc3545;
                color: #721c24;
                font-weight: bold;
                margin: 10px 0;
                text-align: center;
            }
            .hazardous-air {
                background-color: #6f42c1 !important;
                padding: 15px;
                border-radius: 8px;
                border: 3px solid #6f42c1;
                color: white;
                font-weight: bold;
                margin: 10px 0;
                text-align: center;
            }
            .status-message {
                font-size: 20px;
                text-align: center;
                margin: 15px 0;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                border: 2px solid #dee2e6;
                margin: 10px 0;
                text-align: center;
            }
            .forecast-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 12px;
                margin: 15px 0;
                text-align: center;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Initialize Streamlit placeholders
        self.status_label = st.empty()
        self.timer_label = st.empty()
        self.progress_bar = st.empty()
        self.message_placeholder = st.empty()
        self.alert_label = st.empty()
        self.chart_placeholder = st.empty()
        self.forecast_placeholder = st.empty()
        self.bg_container = st.empty()

        # CSV file setup for logging
        self.csv_file = "air_quality_log.csv"
        self.create_csv_file()

    def create_csv_file(self):
        """Create a CSV file to log air quality data."""
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Location", "City Type", "PM2.5", "PM10", "CO", "NO2", "O3", "SO2", "AQI", "Air Quality Status", "Dominant Pollutants", "Session ID"])

    def log_air_quality(self, data):
        """Log air quality data to the CSV file."""
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now(),
                data['location'],
                data['city_type'],
                data['pm25'],
                data['pm10'],
                data['co'],
                data['no2'],
                data['o3'],
                data.get('so2', 0),
                data['aqi'],
                data['status'],
                ', '.join(data.get('dominant_pollutants', [])),
                self.session_id
            ])

    def start_monitoring(self):
        self.is_monitoring = True
        self.session_id = self.create_session()
        self.start_time = time.time()
        self.status_label.text("Status: Air Quality Monitoring Active")
        self.run_monitoring()

    def create_session(self):
        conn = sqlite3.connect('air_quality.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS sessions 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                          start_time TIMESTAMP, 
                          total_alerts INTEGER)''')
        cursor.execute('''INSERT INTO sessions (start_time, total_alerts) VALUES (?, 0)''', (datetime.now(),))
        conn.commit()
        conn.close()
        return cursor.lastrowid

    def get_aqi_status(self, aqi):
        """Get air quality status based on AQI value."""
        if aqi <= 50:
            return "Good", "good-air"
        elif aqi <= 100:
            return "Moderate", "moderate-air"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups", "poor-air"
        elif aqi <= 200:
            return "Unhealthy", "poor-air"
        elif aqi <= 300:
            return "Very Unhealthy", "hazardous-air"
        else:
            return "Hazardous", "hazardous-air"

    def update_ui_style(self, aqi):
        """Update UI style based on AQI"""
        status, css_class = self.get_aqi_status(aqi)
        self.bg_container.markdown(
            f'<div class="{css_class} status-message">🌬️ Air Quality: {status} (AQI: {aqi})</div>', 
            unsafe_allow_html=True
        )

    def run_monitoring(self):
        if st.button("Stop Monitoring", key=f"stop_monitoring_button_{self.session_id}"):
            self.stop_monitoring()
            self.status_label.text("Status: Monitoring Stopped")
            return

        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        while self.is_monitoring:
            # Check time remaining
            elapsed_time = time.time() - self.start_time
            remaining_time = max(0, self.monitoring_duration - elapsed_time)
            
            # Format remaining time as HH:MM:SS
            hours = int(remaining_time // 3600)
            minutes = int((remaining_time % 3600) // 60)
            seconds = int(remaining_time % 60)
            self.timer_label.text(f"Monitoring Time Remaining: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Stop monitoring if time is up
            if remaining_time <= 0:
                st.warning("Monitoring period completed!")
                self.stop_monitoring()
                self.status_label.text("Status: Monitoring Completed")
                return

            # Get current air quality data for selected location
            if self.selected_location:
                self.sensor.set_location(self.selected_location)
            current_data = self.sensor.get_air_quality()
            
            # Log the data
            self.log_air_quality(current_data)
            
            # Display location information
            st.markdown("### 📍 Location Information")
            location_info = current_data.get('location_info', {})
            
            # Highlight the selected location
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px; border-radius: 12px; text-align: center; margin: 10px 0;">
                <h2>🌍 Monitoring: {current_data['location']}</h2>
                <p><strong>City Type:</strong> {current_data['city_type'].title()}</p>
                <p><strong>Coordinates:</strong> {current_data['coordinates']['lat']:.4f}, {current_data['coordinates']['lon']:.4f}</p>
                <p><strong>Elevation:</strong> {current_data['coordinates']['elevation']}m</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.info(f"**Weather Conditions:** {current_data['weather'].title()}")
                st.info(f"**Traffic Intensity:** {current_data['traffic'].title()}")
            with col_info2:
                st.info(f"**Current AQI:** {current_data['aqi']:.0f}")
                st.info(f"**Air Quality Status:** {current_data['status']}")
            
            # Display dominant pollutants for this location
            st.markdown(f"### 🧪 Dominant Pollutants in {current_data['location']}")
            dominant_pollutants = current_data.get('dominant_pollutants', [])
            primary_sources = current_data.get('primary_sources', [])
            
            col_poll1, col_poll2 = st.columns(2)
            with col_poll1:
                st.markdown("**🏭 Primary Pollution Sources:**")
                for source in primary_sources:
                    st.markdown(f"• {source}")
            
            with col_poll2:
                st.markdown("**⚠️ Dominant Pollutants:**")
                for pollutant in dominant_pollutants:
                    st.markdown(f"• {pollutant}")
            
            # Update UI with current metrics
            st.markdown("### 📊 Current Air Quality Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("PM2.5 (μg/m³)", f"{current_data['pm25']:.1f}")
            with col2:
                st.metric("PM10 (μg/m³)", f"{current_data['pm10']:.1f}")
            with col3:
                st.metric("AQI", f"{current_data['aqi']:.0f}")
            with col4:
                st.metric("SO2 (ppb)", f"{current_data.get('so2', 0):.1f}")
            
            # Update status based on AQI
            self.update_ui_style(current_data['aqi'])
            
            # Display pollution sources analysis
            st.markdown(f"### 🔍 Pollution Sources Analysis for {current_data['location']}")
            pollution_sources = current_data.get('pollution_sources', {})
            seasonal_factors = current_data.get('seasonal_factors', {})
            
            col_sources1, col_sources2 = st.columns(2)
            with col_sources1:
                st.markdown("**📈 Pollution Source Intensity:**")
                for source, intensity in pollution_sources.items():
                    color = "🟢" if intensity == "Low" else "🟡" if intensity == "Moderate" else "🔴"
                    st.markdown(f"{color} **{source.title()}:** {intensity}")
            
            with col_sources2:
                st.markdown("**🌤️ Seasonal Factors:**")
                for season, factor in seasonal_factors.items():
                    st.markdown(f"• **{season.title()}:** {factor}")
            
            # Check for alerts
            if current_data['aqi'] > 100:
                self.alert_count += 1
                st.warning(f"⚠️ Air Quality Alert: AQI is {current_data['aqi']:.0f} - {current_data['status']}")
            
            # Generate and display forecast
            forecast_data = self.forecaster.predict_next_hours(current_data, hours=6)
            self.display_forecast(forecast_data)
            
            # Generate and display policy recommendations
            policy_recommendations = self.forecaster.generate_policy_recommendations(current_data, forecast_data)
            self.display_policy_recommendations(policy_recommendations)
            
            # Create real-time chart
            self.create_air_quality_chart(current_data)
            
            time.sleep(2)  # Update every 2 seconds

    def display_forecast(self, forecast_data):
        """Display air quality forecast."""
        st.subheader("🌤️ 6-Hour Air Quality Forecast")
        
        # Create forecast chart
        hours = list(range(1, 7))
        aqi_values = [forecast['aqi'] for forecast in forecast_data]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=aqi_values,
            mode='lines+markers',
            name='Predicted AQI',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Air Quality Index Forecast",
            xaxis_title="Hours from now",
            yaxis_title="AQI",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast summary
        avg_aqi = np.mean(aqi_values)
        status, _ = self.get_aqi_status(avg_aqi)
        
        st.markdown(f"""
        <div class="forecast-card">
            <h3>📊 Forecast Summary</h3>
            <p>Average AQI over next 6 hours: <strong>{avg_aqi:.0f}</strong></p>
            <p>Expected Air Quality: <strong>{status}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    def create_air_quality_chart(self, current_data):
        """Create real-time air quality chart."""
        st.subheader("📈 Real-time Air Quality Metrics")
        
        # Create gauge charts for key pollutants
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # PM2.5 Gauge
            fig_pm25 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_data['pm25'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "PM2.5 (μg/m³)"},
                delta={'reference': 12},
                gauge={
                    'axis': {'range': [None, 50]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 12], 'color': "lightgray"},
                        {'range': [12, 35.4], 'color': "yellow"},
                        {'range': [35.4, 50], 'color': "orange"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 35.4
                    }
                }
            ))
            fig_pm25.update_layout(height=300)
            st.plotly_chart(fig_pm25, use_container_width=True)
        
        with col2:
            # AQI Gauge
            fig_aqi = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_data['aqi'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Air Quality Index"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 300]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "green"},
                        {'range': [51, 100], 'color': "yellow"},
                        {'range': [101, 150], 'color': "orange"},
                        {'range': [151, 200], 'color': "red"},
                        {'range': [201, 300], 'color': "purple"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))
            fig_aqi.update_layout(height=300)
            st.plotly_chart(fig_aqi, use_container_width=True)
        
        with col3:
            # SO2 Gauge
            fig_so2 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_data.get('so2', 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "SO2 (ppb)"},
                delta={'reference': 5},
                gauge={
                    'axis': {'range': [None, 20]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgray"},
                        {'range': [5, 10], 'color': "yellow"},
                        {'range': [10, 20], 'color': "orange"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 10
                    }
                }
            ))
            fig_so2.update_layout(height=300)
            st.plotly_chart(fig_so2, use_container_width=True)

    def display_policy_recommendations(self, recommendations):
        """Display AI-generated policy recommendations."""
        st.subheader("🤖 AI-Powered Policy Recommendations")
        
        # Priority level indicator
        priority_colors = {
            'critical': '#dc3545',
            'high': '#fd7e14',
            'medium': '#ffc107',
            'low': '#28a745'
        }
        
        priority_color = priority_colors.get(recommendations['priority_level'], '#6c757d')
        
        st.markdown(f"""
        <div style="background-color: {priority_color}; color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h4>Priority Level: {recommendations['priority_level'].upper()}</h4>
            <p>Confidence Score: {recommendations['confidence_score']:.1%}</p>
            <p>Economic Impact: {recommendations['economic_impact'].title()}</p>
            <p>Implementation Timeline: {recommendations['implementation_timeline']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Emergency actions
        if recommendations['emergency_actions']:
            st.markdown("### 🚨 Emergency Actions Required")
            for action in recommendations['emergency_actions']:
                st.markdown(f"• **{action}**")
        
        # AI Insights
        if recommendations.get('ai_insights'):
            st.markdown("### 🧠 AI Insights")
            for insight in recommendations['ai_insights']:
                st.info(insight)
        
        # Short-term policies
        if recommendations['short_term_policies']:
            st.markdown("### 📋 Short-term Policy Recommendations")
            col1, col2 = st.columns(2)
            
            mid_point = len(recommendations['short_term_policies']) // 2
            with col1:
                for policy in recommendations['short_term_policies'][:mid_point]:
                    st.markdown(f"• {policy}")
            with col2:
                for policy in recommendations['short_term_policies'][mid_point:]:
                    st.markdown(f"• {policy}")
        
        # Long-term policies
        if recommendations['long_term_policies']:
            st.markdown("### 🏗️ Long-term Policy Recommendations")
            col1, col2 = st.columns(2)
            
            mid_point = len(recommendations['long_term_policies']) // 2
            with col1:
                for policy in recommendations['long_term_policies'][:mid_point]:
                    st.markdown(f"• {policy}")
            with col2:
                for policy in recommendations['long_term_policies'][mid_point:]:
                    st.markdown(f"• {policy}")
        
        # Policy impact analysis
        st.markdown("### 📊 Policy Impact Analysis")
        
        impact_data = {
            'Category': ['Health Impact', 'Economic Cost', 'Implementation Time', 'Public Acceptance'],
            'Score': [
                recommendations['confidence_score'] * 100,
                random.uniform(60, 90) if recommendations['economic_impact'] == 'high' else random.uniform(30, 70),
                random.uniform(20, 80),
                random.uniform(50, 85)
            ]
        }
        
        impact_df = pd.DataFrame(impact_data)
        
        fig_impact = px.bar(impact_df, x='Category', y='Score', 
                           title="Policy Implementation Impact Scores",
                           color='Score', color_continuous_scale='RdYlGn')
        fig_impact.update_layout(height=300)
        st.plotly_chart(fig_impact, use_container_width=True)

    def stop_monitoring(self):
        self.is_monitoring = False
        self.chart_placeholder.empty()
        self.forecast_placeholder.empty()

def main():
    st.set_page_config(
        page_title="Air Pollution Forecaster",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌬️ Air Pollution Forecasting System")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Location selection
    location = st.sidebar.selectbox(
        "Select Location",
        ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego"]
    )
    
    # Monitoring duration
    duration_hours = st.sidebar.slider("Monitoring Duration (hours)", 1, 48, 24)
    
    # Alert threshold
    alert_threshold = st.sidebar.slider("Alert Threshold (AQI)", 50, 200, 100)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 About")
    st.sidebar.markdown("""
    This system monitors air quality in real-time and provides:
    - **Real-time AQI monitoring**
    - **6-hour pollution forecasts**
    - **Air quality alerts**
    - **Historical data logging**
    """)
    
    if 'monitoring_started' not in st.session_state:
        st.session_state.monitoring_started = False

    if st.button("🚀 Start Air Quality Monitoring", key="start_monitoring_button", use_container_width=True):
        st.session_state.monitoring_started = True
        forecaster = AirPollutionForecaster()
        forecaster.monitoring_duration = duration_hours * 3600  # Convert to seconds
        forecaster.selected_location = location  # Set the selected location
        forecaster.start_monitoring()

if __name__ == "__main__":
    main()