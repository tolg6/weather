import streamlit as st
import utils.db_connector
from dotenv import load_dotenv
import os
import pandas as pd
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta


### DB Connection ###
load_dotenv()
DB_CONFIG = {
    "name": os.getenv("DB_NAME"),
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT")
}

def get_db_connection():
    return utils.db_connector.Connector(connection_params=DB_CONFIG).connect()


### Session State ###
def init_session_state():
    """session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'device_id' not in st.session_state:
        st.session_state.device_id = None

def authenticate_user(username: str, password: str) -> bool:
    """User authentication"""
    conn = get_db_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()
        
        # search customer 
        query = "SELECT device_name, device_id FROM public.devices WHERE device_name = %s AND device_id = %s"
        cursor.execute(query, (username, password))
        result = cursor.fetchone()
        
        if result is not None:
            # add device_id to session state
            st.session_state.device_id = result[1]
            cursor.close()
            conn.close()
            return True
        else:
            cursor.close()
            conn.close()
            return False
        
    except psycopg2.Error as e:
        st.error("Something went wrong. Try again later.")
        if conn:
            conn.close()
        return False

def show_login_page():
    """Login Page"""
    st.title("🔐 Giriş Yap")
    
    with st.form("login_form"):
        username = st.text_input("Kullanıcı Adı")
        password = st.text_input("Şifre", type="password")
        submit_button = st.form_submit_button("Giriş Yap")
        
        if submit_button:
            if username and password:
                if authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Giriş başarılı! Yönlendiriliyorsunuz...")
                    st.rerun()
                else:
                    st.error("Hatalı kullanıcı adı veya şifre!")
            else:
                st.warning("Lütfen tüm alanları doldurun!")

def logout():
    """Logout"""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.device_id = None
    st.rerun()


# Load Predicted Data
def load_predicted_data():
    conn = get_db_connection()
    if conn is None:
        st.error("Veritabanı bağlantısı kurulamadı")
        return pd.DataFrame()
    
    try:
        query = """
        SELECT timestamp, temp_prediction 
        FROM public.predictions 
        where device_id = %s
        ORDER BY timestamp
        """
        
        df = pd.read_sql(query, conn, params=[st.session_state.device_id])
        conn.close()
        return df
        
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

def load_weather_data():
    conn = get_db_connection()
    if conn is None:
        st.error("Veritabanı bağlantısı kurulamadı")
        return pd.DataFrame()
    
    try:
        #device idye göre filtrele
        query = """
        SELECT timestamp, temperature, humidity, pressure, wind_speed, 
               wind_direction, beaufort, direction_code 
        FROM public.weather_data_0001 
        where id = %s
        ORDER BY timestamp
        """
        
        df = pd.read_sql(query, conn, params=[st.session_state.device_id])
        conn.close()
        return df
        
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

def process_to_hourly(df):
    """5 dakikalık verileri saatlik ortalamaya çevirir"""
    if df.empty:
        return df
    
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy.set_index('timestamp', inplace=True)
    
    # Saatlik ortalama hesapla
    hourly_data = df_copy.resample('H').agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'pressure': 'mean',
        'wind_speed': 'mean',
        'wind_direction': 'mean',
        'beaufort': 'mean'
    }).round(2)
    
    hourly_data.reset_index(inplace=True)
    return hourly_data

def create_metric_card(title, value, unit, icon="📊"):
    """Özel metrik kartı oluşturur"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    ">
        <div style="font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; opacity: 0.9;">
            {icon} {title}
        </div>
        <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem;">
            {value}
        </div>
        <div style="font-size: 0.8rem; opacity: 0.8;">
            {unit}
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_comparison_card(title, today_val, yesterday_val, unit, icon="📊"):
    """Bugün ve dün karşılaştırmalı metrik kartı"""
    if yesterday_val != 0 and not pd.isna(yesterday_val):
        change = ((today_val - yesterday_val) / yesterday_val) * 100
    else:
        change = 0
    arrow = "🔼" if change > 0 else ("🔽" if change < 0 else "➡️")
    change_text = f"{arrow} {change:.1f}%"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
        padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 0.5rem 0;">
        <div style="font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; opacity: 0.9;">
            {icon} {title}
        </div>
        <div style="font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem;">
            {today_val:.1f} {unit}
        </div>
        <div style="font-size: 0.8rem; opacity: 0.8;">
            Dün: {yesterday_val:.1f} {unit} | {change_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

def wind_direction_to_text(degrees):
    """Rüzgar yönünü derece cinsinden metne çevirir"""
    if pd.isna(degrees):
        return "N/A"
    directions = ["K", "KKD", "KD", "DKD", "D", "DGD", "GD", "GGD", 
                  "G", "GGB", "GB", "BGB", "B", "BBK", "BK", "KBK"]
    idx = round(degrees / 22.5) % 16
    return directions[idx]

def main():
    st.set_page_config(
        page_title="Hava Durumu Dashboard", 
        page_icon="🌤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS ile özel stiller
    st.markdown("""
    <style>
        .stSelectbox > div > div > div {
            background-color: #f8f9fa;
        }
        .metric-row {
            display: flex;
            gap: 1rem;
            margin: 1rem 0;
        }
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Session state'i başlat
    init_session_state()
    
    # Login kontrolu
    if not st.session_state.logged_in:
        show_login_page()
        st.stop()
    
    # Ana başlık
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0;">🌤️ Hava Durumu Dashboard</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Hoş Geldiniz, {st.session_state.username}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Yan panel ve çıkış butonu
    with st.sidebar:
        st.header("⚙️ Kontrol Paneli")
        st.write(f"**Aktif Cihaz:** {st.session_state.device_id}")
        
        # Çıkış butonu
        if st.button("🚪 Çıkış Yap", use_container_width=True):
            logout()
        
        st.markdown("---")
        
        # Veri türü seçimi
        show_hourly = st.checkbox("Saatlik Ortalama Göster", value=True)
        
        # Veri yenileme butonu
        if st.button("🔄 Verileri Yenile", use_container_width=True):
            st.cache_data.clear()
            st.success("Veriler yenilendi!")
    
    # Veri yükleme
    with st.spinner("Veriler yükleniyor..."):
        df = load_weather_data()
    
    if df.empty:
        st.warning("Bu kullanıcı için veri bulunamadı.")
        return
    
    # Saatlik veri işleme
    if show_hourly:
        display_df = process_to_hourly(df)
        data_type_text = "Saatlik Ortalama"
    else:
        display_df = df
        data_type_text = "Ham Veri"
    
    st.subheader(f"📊 {data_type_text} Verileri")
    
    # Özet kartları
    if not display_df.empty:
        st.markdown("### 📈 Güncel Durum")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            latest_temp = display_df['temperature'].iloc[-1] if not display_df['temperature'].empty else 0
            create_metric_card("Sıcaklık", f"{latest_temp:.1f}", "°C", "🌡️")
        
        with col2:
            latest_humidity = display_df['humidity'].iloc[-1] if not display_df['humidity'].empty else 0
            create_metric_card("Nem", f"{latest_humidity:.1f}", "%", "💧")
        
        with col3:
            latest_pressure = display_df['pressure'].iloc[-1] if not display_df['pressure'].empty else 0
            create_metric_card("Basınç", f"{latest_pressure:.1f}", "hPa", "🌪️")
        
        with col4:
            latest_wind_speed = display_df['wind_speed'].iloc[-1] if not display_df['wind_speed'].empty else 0
            create_metric_card("Rüzgar Hızı", f"{latest_wind_speed:.1f}", "m/s", "💨")
        
        with col5:
            latest_wind_dir = display_df['wind_direction'].iloc[-1] if not display_df['wind_direction'].empty else 0
            wind_text = wind_direction_to_text(latest_wind_dir)
            create_metric_card("Rüzgar Yönü", f"{wind_text}", f"({latest_wind_dir:.0f}°)", "🧭")
        
        st.markdown("---")

        # Günlük karşılaştırmalı kartlar
        st.markdown("### 📊 Günlük Karşılaştırma")
        if 'timestamp' in display_df.columns:
            display_df['date'] = display_df['timestamp'].dt.date
            daily_means = display_df.groupby('date')[['temperature','humidity','pressure']].mean()

            if len(daily_means) >= 2:
                today_vals = daily_means.iloc[-1]
                yesterday_vals = daily_means.iloc[-2]

                c1, c2, c3 = st.columns(3)
                with c1:
                    create_comparison_card("Sıcaklık", today_vals['temperature'], yesterday_vals['temperature'], "°C", "🌡️")
                with c2:
                    create_comparison_card("Nem", today_vals['humidity'], yesterday_vals['humidity'], "%", "💧")
                with c3:
                    create_comparison_card("Basınç", today_vals['pressure'], yesterday_vals['pressure'], "hPa", "🌪️")

        st.markdown("---")
        
        # Ana grafikler
        st.markdown("### 📈 Zaman Serisi Grafikleri")
        
        # Sıcaklık ve Nem
        col1, col2 = st.columns(2)
        
        with col1:
            if 'timestamp' in display_df.columns and 'temperature' in display_df.columns:
                fig_temp = px.line(
                    display_df, 
                    x='timestamp', 
                    y='temperature',
                    title='Sıcaklık Değişimi',
                    labels={'temperature': 'Sıcaklık (°C)', 'timestamp': 'Zaman'}
                )
                fig_temp.update_traces(line_color='#FF6B6B', line_width=3)
                fig_temp.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12)
                )
                st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            if 'timestamp' in display_df.columns and 'humidity' in display_df.columns:
                fig_humidity = px.line(
                    display_df, 
                    x='timestamp', 
                    y='humidity',
                    title='Nem Oranı Değişimi',
                    labels={'humidity': 'Nem (%)', 'timestamp': 'Zaman'}
                )
                fig_humidity.update_traces(line_color='#4ECDC4', line_width=3)
                fig_humidity.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12)
                )
                st.plotly_chart(fig_humidity, use_container_width=True)

        # Predicted Temperature and Actual Temperature Comparison without merging
        if 'timestamp' in display_df.columns and 'temperature' in display_df.columns:
            predicted_df = load_predicted_data()
            if not predicted_df.empty:
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=display_df['timestamp'], 
                    y=display_df['temperature'], 
                    mode='lines', 
                    name='Gerçek Sıcaklık',
                    line=dict(color='#FF6B6B', width=3)
                ))
                fig_pred.add_trace(go.Scatter(
                    x=predicted_df['timestamp'], 
                    y=predicted_df['temp_prediction'], 
                    mode='lines', 
                    name='Tahmin Edilen Sıcaklık',
                    line=dict(color='#1E90FF', width=3)
                ))
                fig_pred.update_layout(
                    title='Gerçek ve Tahmin Edilen Sıcaklık Karşılaştırması',
                    xaxis_title='Zaman',
                    yaxis_title='Sıcaklık (°C)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12)
                )
                st.plotly_chart(fig_pred, use_container_width=True)
        
        # Basınç ve Rüzgar Hızı
        col1, col2 = st.columns(2)
        
        with col1:
            if 'timestamp' in display_df.columns and 'pressure' in display_df.columns:
                fig_pressure = px.area(
                    display_df, 
                    x='timestamp', 
                    y='pressure',
                    title='Atmosfer Basıncı',
                    labels={'pressure': 'Basınç (hPa)', 'timestamp': 'Zaman'}
                )
                fig_pressure.update_traces(fill='tonexty', fillcolor='rgba(255, 193, 7, 0.3)', line_color='#FFC107')
                fig_pressure.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12)
                )
                st.plotly_chart(fig_pressure, use_container_width=True)
        
        with col2:
            if 'timestamp' in display_df.columns and 'wind_speed' in display_df.columns:
                fig_wind_speed = px.line(
                    display_df, 
                    x='timestamp', 
                    y='wind_speed',
                    title='Rüzgar Hızı',
                    labels={'wind_speed': 'Rüzgar Hızı (m/s)', 'timestamp': 'Zaman'}
                )
                fig_wind_speed.update_traces(line_color='#9B59B6', line_width=3)
                fig_wind_speed.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12)
                )
                st.plotly_chart(fig_wind_speed, use_container_width=True)
        
        st.markdown("---")

        # İki eksenli grafikler
        st.markdown("### 📊 İlişkiler (Çift Ekseni Grafikler)")

        # Sıcaklık - Nem
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Scatter(x=display_df['timestamp'], y=display_df['temperature'], name="Sıcaklık (°C)", line=dict(color="red")), secondary_y=False)
        fig1.add_trace(go.Scatter(x=display_df['timestamp'], y=display_df['humidity'], name="Nem (%)", line=dict(color="blue")), secondary_y=True)
        fig1.update_layout(title="Sıcaklık - Nem İlişkisi")
        st.plotly_chart(fig1, use_container_width=True)

        # Sıcaklık - Basınç
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=display_df['timestamp'], y=display_df['temperature'], name="Sıcaklık (°C)", line=dict(color="red")), secondary_y=False)
        fig2.add_trace(go.Scatter(x=display_df['timestamp'], y=display_df['pressure'], name="Basınç (hPa)", line=dict(color="orange")), secondary_y=True)
        fig2.update_layout(title="Sıcaklık - Basınç İlişkisi")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        
        # Rüzgar yönü polar grafik (küçültülmüş)
        if 'wind_direction' in display_df.columns:
            st.markdown("### 🧭 Rüzgar Yönü Dağılımı")
            col1, col2, col3 = st.columns([1,2,1])
            with col2:  # ortada küçük şekilde göster
                fig_wind = go.Figure()
                fig_wind.add_trace(go.Scatterpolar(
                    r=[1] * len(display_df),
                    theta=display_df['wind_direction'],
                    mode='markers',
                    marker=dict(size=6, color=display_df['wind_direction'], colorscale='HSV'),
                    name='Rüzgar Yönü'
                ))
                fig_wind.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=False, range=[0, 1.2]),
                        angularaxis=dict(
                            tickmode='array',
                            tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                            ticktext=['K', 'KD', 'D', 'GD', 'G', 'GB', 'B', 'BK']
                        )
                    ),
                    showlegend=False,
                    font=dict(size=11),
                    height=400, width=400
                )
                st.plotly_chart(fig_wind, use_container_width=False)

        st.markdown("---")
        
        # Veri Tablosu
        st.markdown("### 📋 Veri Tablosu")
        st.dataframe(display_df.tail(100), use_container_width=True)

if __name__ == "__main__":
    main()
