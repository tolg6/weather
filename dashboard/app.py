import streamlit as st
import utils.db_connector
from dotenv import load_dotenv
import os
import pandas as pd
import psycopg2


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


def main():
    st.set_page_config(page_title="Dashboard", layout="wide")
    
    # Session state'i başlat
    init_session_state()
    
    # Login kontrolu
    if not st.session_state.logged_in:
        show_login_page()
        st.stop()
    
    # Ana sayfa
    st.title(f"Hoş Geldiniz, {st.session_state.username}")
    
    # Çıkış butonu
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Çıkış Yap"):
            logout()
    
    # Veri yükleme
    with st.spinner("Veriler yükleniyor..."):
        df = load_weather_data()
    
    if df.empty:
        st.warning("Bu kullanıcı için veri bulunamadı.")
        return
    
    # Dashboard
    st.subheader("Dashboard")
    st.dataframe(df)

    # summary stats table
    st.subheader("Summary Statistics")
    st.write(df.describe().T)

    # graph
    if 'timestamp' in df.columns and 'temperature' in df.columns:
        st.subheader("Temperature Over Time")
        st.line_chart(df.set_index('timestamp')['temperature'])
    
    if 'timestamp' in df.columns and 'humidity' in df.columns:
        st.subheader("Humidity Over Time")
        st.line_chart(df.set_index('timestamp')['humidity'])

if __name__ == "__main__":
    main()