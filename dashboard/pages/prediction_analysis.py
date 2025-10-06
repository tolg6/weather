import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import utils.db_connector
from dotenv import load_dotenv
import os

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

def load_predicted_data(device_id):
    conn = get_db_connection()
    if conn is None:
        st.error("Veritabanı bağlantısı kurulamadı")
        return pd.DataFrame()
    
    try:
        query = """
        SELECT model, forecast_datetime as timestamp, predicted_temperature as temp_prediction 
        FROM public.predictions 
        WHERE device_id = %s
        ORDER BY timestamp
        """
        df = pd.read_sql(query, conn, params=[device_id])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        conn.close()
        return df
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

def load_weather_data(device_id):
    conn = get_db_connection()
    if conn is None:
        st.error("Veritabanı bağlantısı kurulamadı")
        return pd.DataFrame()
    
    try:
        query = """
        SELECT timestamp, temperature
        FROM public.weather_data_0001 
        WHERE id = %s
        ORDER BY timestamp
        """
        df = pd.read_sql(query, conn, params=[device_id])
        df['timestamp'] = pd.to_datetime(df['timestamp']) + pd.Timedelta(hours=3)
        df.set_index('timestamp').resample("H").mean().reset_index()
        conn.close()
        return df
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

def calculate_smape(actual, predicted):
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    diff = np.abs(actual - predicted) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

def calculate_mape(actual, predicted):
    """Mean Absolute Percentage Error"""
    mask = actual != 0
    return 100 * np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))

def calculate_metrics(actual, predicted):
    """Tüm hata metriklerini hesapla"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    smape = calculate_smape(actual, predicted)
    try:
        mape = calculate_mape(actual, predicted)
    except:
        mape = np.nan
    r2 = r2_score(actual, predicted)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'SMAPE': smape,
        'MAPE': mape,
        'R²': r2
    }

def create_metric_card_analysis(title, value, unit="", description="", color="#667eea"):
    """Analiz sayfası için özel metrik kartı"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color} 0%, {color}dd 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">
        <div style="font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; opacity: 0.9;">
            {title}
        </div>
        <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
            {value}{unit}
        </div>
        <div style="font-size: 0.75rem; opacity: 0.85;">
            {description}
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_prediction_analysis():
    st.set_page_config(
        page_title="Tahmin Analizi", 
        page_icon="🤖",
        layout="wide"
    )
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
        <h1 style="margin: 0;">🤖 Tahmin Performans Analizi</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Model performansı ve hata metrikleri</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'device_id' not in st.session_state or st.session_state.device_id is None:
        st.error("Lütfen önce giriş yapın!")
        return
    
    device_id = st.session_state.device_id
    
    # Verileri yükle
    with st.spinner("Veriler yükleniyor..."):
        pred_df = load_predicted_data(device_id)
        actual_df = load_weather_data(device_id)
    
    if pred_df.empty or actual_df.empty:
        st.warning("Yeterli veri bulunamadı.")
        return
    
    # Model seçimi
    models = pred_df['model'].unique()
    selected_model = st.selectbox("📊 Model Seçin", models, key="model_select")
    
    # Seçili model için veri filtrele
    model_pred = pred_df[pred_df['model'] == selected_model].copy()
    
    # Timestamp üzerinden birleştirme (en yakın timestamp eşleştirmesi)
    model_pred = model_pred.sort_values('timestamp')
    actual_df = actual_df.sort_values('timestamp')
    
    # Merge with tolerance (5 minute tolerance)
    merged_df = pd.merge_asof(
        model_pred,
        actual_df[['timestamp', 'temperature']],
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('5min')
    )
    
    merged_df = merged_df.dropna(subset=['temperature', 'temp_prediction'])
    
    if merged_df.empty:
        st.warning("Eşleşen veri bulunamadı. Zaman aralıklarını kontrol edin.")
        return
    
    # Hata hesaplama
    merged_df['error'] = merged_df['temperature'] - merged_df['temp_prediction']
    merged_df['abs_error'] = np.abs(merged_df['error'])
    merged_df['squared_error'] = merged_df['error'] ** 2
    merged_df['percentage_error'] = (merged_df['error'] / merged_df['temperature']) * 100
    
    # Metrikleri hesapla
    metrics = calculate_metrics(merged_df['temperature'].values, merged_df['temp_prediction'].values)
    
    st.markdown("---")
    
    # Metrik kartları
    st.markdown("### 📊 Performans Metrikleri")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        create_metric_card_analysis(
            "MAE", 
            f"{metrics['MAE']:.3f}", 
            "°C",
            "Ortalama Mutlak Hata",
            "#FF6B6B"
        )
    
    with col2:
        create_metric_card_analysis(
            "RMSE", 
            f"{metrics['RMSE']:.3f}", 
            "°C",
            "Kök Ortalama Kare Hata",
            "#4ECDC4"
        )
    
    with col3:
        create_metric_card_analysis(
            "SMAPE", 
            f"{metrics['SMAPE']:.2f}", 
            "%",
            "Simetrik MAPE",
            "#FFC107"
        )
    
    with col4:
        if not np.isnan(metrics['MAPE']):
            create_metric_card_analysis(
                "MAPE", 
                f"{metrics['MAPE']:.2f}", 
                "%",
                "Ortalama Mutlak Yüzde Hata",
                "#9B59B6"
            )
        else:
            create_metric_card_analysis(
                "MAPE", 
                "N/A", 
                "",
                "Hesaplanamadı",
                "#9B59B6"
            )
    
    with col5:
        create_metric_card_analysis(
            "R² Score", 
            f"{metrics['R²']:.4f}", 
            "",
            "Belirleme Katsayısı",
            "#43cea2"
        )
    
    st.markdown("---")
    
    # Ana karşılaştırma grafiği
    st.markdown("### 📈 Gerçek vs Tahmin Karşılaştırması")
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Scatter(
        x=merged_df['timestamp'],
        y=merged_df['temperature'],
        mode='lines',
        name='Gerçek Sıcaklık',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig_comparison.add_trace(go.Scatter(
        x=merged_df['timestamp'],
        y=merged_df['temp_prediction'],
        mode='lines',
        name='Tahmin',
        line=dict(color='#4ECDC4', width=2, dash='dash')
    ))
    
    fig_comparison.update_layout(
        title=f'{selected_model} - Gerçek vs Tahmin',
        xaxis_title='Zaman',
        yaxis_title='Sıcaklık (°C)',
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.markdown("---")
    
    # İki sütunlu grafikler
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Scatter Plot - Gerçek vs Tahmin")
        fig_scatter = px.scatter(
            merged_df,
            x='temperature',
            y='temp_prediction',
            trendline='ols',
            labels={'temperature': 'Gerçek Sıcaklık (°C)', 'temp_prediction': 'Tahmin (°C)'},
            color='abs_error',
            color_continuous_scale='Reds'
        )
        
        # Ideal çizgi ekle
        min_val = min(merged_df['temperature'].min(), merged_df['temp_prediction'].min())
        max_val = max(merged_df['temperature'].max(), merged_df['temp_prediction'].max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='İdeal Çizgi',
            line=dict(color='green', dash='dash')
        ))
        
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.markdown("### 📉 Hata Dağılımı")
        fig_error_dist = px.histogram(
            merged_df,
            x='error',
            nbins=50,
            labels={'error': 'Hata (°C)', 'count': 'Frekans'},
            color_discrete_sequence=['#667eea']
        )
        
        # Normal dağılım eğrisi ekle
        mean_error = merged_df['error'].mean()
        std_error = merged_df['error'].std()
        x_range = np.linspace(merged_df['error'].min(), merged_df['error'].max(), 100)
        y_normal = ((1 / (std_error * np.sqrt(2 * np.pi))) * 
                   np.exp(-0.5 * ((x_range - mean_error) / std_error) ** 2))
        y_normal = y_normal * len(merged_df) * (merged_df['error'].max() - merged_df['error'].min()) / 50
        
        fig_error_dist.add_trace(go.Scatter(
            x=x_range,
            y=y_normal,
            mode='lines',
            name='Normal Dağılım',
            line=dict(color='red', width=2)
        ))
        
        fig_error_dist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig_error_dist, use_container_width=True)
    
    st.markdown("---")
    
    # Zaman serisi hata analizi
    st.markdown("### 📈 Zaman Serisi Hata Analizi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Mutlak Hata Zaman Serisi")
        fig_abs_error = px.line(
            merged_df,
            x='timestamp',
            y='abs_error',
            labels={'abs_error': 'Mutlak Hata (°C)', 'timestamp': 'Zaman'}
        )
        fig_abs_error.add_hline(
            y=merged_df['abs_error'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Ortalama: {merged_df['abs_error'].mean():.3f}°C"
        )
        fig_abs_error.update_traces(line_color='#FF6B6B')
        fig_abs_error.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350
        )
        st.plotly_chart(fig_abs_error, use_container_width=True)
    
    with col2:
        st.markdown("#### Hata (Gerçek - Tahmin)")
        fig_error = px.line(
            merged_df,
            x='timestamp',
            y='error',
            labels={'error': 'Hata (°C)', 'timestamp': 'Zaman'}
        )
        fig_error.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_error.add_hline(
            y=merged_df['error'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Bias: {merged_df['error'].mean():.3f}°C"
        )
        fig_error.update_traces(line_color='#4ECDC4')
        fig_error.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350
        )
        st.plotly_chart(fig_error, use_container_width=True)
    
    st.markdown("---")
    
    # Hata istatistikleri
    st.markdown("### 📊 Hata İstatistikleri")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ortalama Hata (Bias)", f"{merged_df['error'].mean():.3f}°C")
        st.metric("Medyan Hata", f"{merged_df['error'].median():.3f}°C")
    
    with col2:
        st.metric("Standart Sapma", f"{merged_df['error'].std():.3f}°C")
        st.metric("Varyans", f"{merged_df['error'].var():.3f}")
    
    with col3:
        st.metric("Min Hata", f"{merged_df['error'].min():.3f}°C")
        st.metric("Max Hata", f"{merged_df['error'].max():.3f}°C")
    
    with col4:
        st.metric("Q1 (25%)", f"{merged_df['error'].quantile(0.25):.3f}°C")
        st.metric("Q3 (75%)", f"{merged_df['error'].quantile(0.75):.3f}°C")
    
    st.markdown("---")
    
    # Hata yüzdeleri analizi
    st.markdown("### 📊 Hata Aralıkları Analizi")
    
    # Farklı hata aralıklarında tahmin sayısı
    error_ranges = {
        '< 0.5°C': (merged_df['abs_error'] < 0.5).sum(),
        '0.5-1°C': ((merged_df['abs_error'] >= 0.5) & (merged_df['abs_error'] < 1)).sum(),
        '1-2°C': ((merged_df['abs_error'] >= 1) & (merged_df['abs_error'] < 2)).sum(),
        '2-3°C': ((merged_df['abs_error'] >= 2) & (merged_df['abs_error'] < 3)).sum(),
        '> 3°C': (merged_df['abs_error'] >= 3).sum()
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_error_ranges = go.Figure(data=[
            go.Bar(
                x=list(error_ranges.keys()),
                y=list(error_ranges.values()),
                marker_color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#c0392b']
            )
        ])
        fig_error_ranges.update_layout(
            title='Hata Aralıklarına Göre Dağılım',
            xaxis_title='Hata Aralığı',
            yaxis_title='Tahmin Sayısı',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig_error_ranges, use_container_width=True)
    
    with col2:
        error_ranges_pct = {k: (v/len(merged_df))*100 for k, v in error_ranges.items()}
        fig_error_pie = go.Figure(data=[
            go.Pie(
                labels=list(error_ranges_pct.keys()),
                values=list(error_ranges_pct.values()),
                marker=dict(colors=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#c0392b']),
                textinfo='label+percent'
            )
        ])
        fig_error_pie.update_layout(
            title='Hata Aralıkları Yüzde Dağılımı',
            height=400
        )
        st.plotly_chart(fig_error_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Rezidual analizi
    st.markdown("### 📊 Rezidual (Artık) Analizi")
    
    fig_residuals = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Rezidual vs Tahmin', 'Rezidual vs Zaman', 
                       'Q-Q Plot', 'Rezidual Otokorelasyon'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Rezidual vs Tahmin
    fig_residuals.add_trace(
        go.Scatter(x=merged_df['temp_prediction'], y=merged_df['error'],
                  mode='markers', marker=dict(color='#667eea', size=5),
                  showlegend=False),
        row=1, col=1
    )
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Rezidual vs Zaman
    fig_residuals.add_trace(
        go.Scatter(x=merged_df['timestamp'], y=merged_df['error'],
                  mode='lines+markers', marker=dict(color='#4ECDC4', size=3),
                  line=dict(width=1), showlegend=False),
        row=1, col=2
    )
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    
    # Q-Q Plot
    sorted_residuals = np.sort(merged_df['error'])
    theoretical_quantiles = np.linspace(-3, 3, len(sorted_residuals))
    theoretical_quantiles = theoretical_quantiles * merged_df['error'].std() + merged_df['error'].mean()
    
    fig_residuals.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sorted_residuals,
                  mode='markers', marker=dict(color='#FF6B6B', size=5),
                  showlegend=False),
        row=2, col=1
    )
    min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
    max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
    fig_residuals.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                  mode='lines', line=dict(color='red', dash='dash'),
                  showlegend=False),
        row=2, col=1
    )
    
    # Otokorelasyon
    from pandas.plotting import autocorrelation_plot
    acf_values = [merged_df['error'].autocorr(lag=i) for i in range(min(50, len(merged_df)//2))]
    
    fig_residuals.add_trace(
        go.Bar(x=list(range(len(acf_values))), y=acf_values,
              marker=dict(color='#9B59B6'), showlegend=False),
        row=2, col=2
    )
    fig_residuals.add_hline(y=0, line_dash="solid", line_color="black", row=2, col=2)
    
    fig_residuals.update_xaxes(title_text="Tahmin (°C)", row=1, col=1)
    fig_residuals.update_xaxes(title_text="Zaman", row=1, col=2)
    fig_residuals.update_xaxes(title_text="Teorik Kantiller", row=2, col=1)
    fig_residuals.update_xaxes(title_text="Lag", row=2, col=2)
    
    fig_residuals.update_yaxes(title_text="Rezidual", row=1, col=1)
    fig_residuals.update_yaxes(title_text="Rezidual", row=1, col=2)
    fig_residuals.update_yaxes(title_text="Örnek Kantiller", row=2, col=1)
    fig_residuals.update_yaxes(title_text="Otokorelasyon", row=2, col=2)
    
    fig_residuals.update_layout(
        height=800,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_residuals, use_container_width=True)
    
    st.markdown("---")
    
    # Detaylı veri tablosu
    st.markdown("### 📋 Detaylı Tahmin Verileri")
    display_df = merged_df[['timestamp', 'temperature', 'temp_prediction', 'error', 'abs_error', 'percentage_error']].copy()
    display_df.columns = ['Zaman', 'Gerçek (°C)', 'Tahmin (°C)', 'Hata (°C)', 'Mutlak Hata (°C)', 'Yüzde Hata (%)']
    display_df = display_df.round(3)
    
    st.dataframe(display_df.tail(100), use_container_width=True)
    
    # İndir butonu
    csv = merged_df.to_csv(index=False)
    st.download_button(
        label="📥 Analiz Verilerini İndir (CSV)",
        data=csv,
        file_name=f"{selected_model}_analysis_{device_id}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    show_prediction_analysis()