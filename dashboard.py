import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Prediksi Harga Beras Premium Papua",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric > label {
        font-size: 0.8rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌾 Dashboard Prediksi Harga Beras Premium Papua</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Menu Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["🏠 Beranda", "📈 Analisis Data", "🤖 Model Prediksi", "📋 Evaluasi Model"]
)

# Fungsi untuk memuat dan memproses data
@st.cache_data
def load_data():
    df = pd.read_csv('Beras Premium Train.csv')
    df_papua = df[['Date', 'Papua']]
    df_cleaned = df_papua.dropna(axis=0)
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], format='%Y-%m-%d')
    df_cleaned.sort_values(by='Date', inplace=True)
    return df_cleaned

# Fungsi untuk normalisasi data
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data[['Papua']])
    return data_normalized, scaler

# Fungsi untuk membuat sequences
def create_sequences(data, timesteps=60):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:(i+timesteps), 0])
        y.append(data[i+timesteps, 0])
    return np.array(X), np.array(y)

# Load data
try:
    df_cleaned = load_data()
    st.sidebar.success("✅ Data berhasil dimuat!")
    st.sidebar.info(f"📊 Total data: {len(df_cleaned)} hari")
    st.sidebar.info(f"📅 Periode: {df_cleaned['Date'].min().strftime('%d %B %Y')} - {df_cleaned['Date'].max().strftime('%d %B %Y')}")
except Exception as e:
    st.error(f"❌ Error memuat data: {e}")
    st.stop()

# HALAMAN BERANDA
if menu == "🏠 Beranda":
    st.subheader("Selamat Datang di Dashboard Prediksi Harga Beras Premium Papua")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📊 Tentang Dashboard Ini
        Dashboard ini dirancang untuk menganalisis dan memprediksi harga beras premium di Papua menggunakan model LSTM (Long Short-Term Memory).
        
        **Fitur Utama:**
        - 📈 Visualisasi trend harga beras premium
        - 🔍 Analisis statistik mendalam
        - 🤖 Prediksi harga menggunakan AI
        - 📋 Evaluasi performa model
        
        **Cara Menggunakan:**
        1. Pilih menu di sidebar sebelah kiri
        2. Jelajahi analisis data dan visualisasi
        3. Lihat hasil prediksi model
        4. Evaluasi performa model
        """)
    
    with col2:
        st.markdown("### 📊 Statistik Singkat")
        harga_min = df_cleaned['Papua'].min()
        harga_max = df_cleaned['Papua'].max()
        harga_rata = df_cleaned['Papua'].mean()
        
        st.metric("💰 Harga Minimum", f"Rp {harga_min:,.0f}")
        st.metric("📈 Harga Maksimum", f"Rp {harga_max:,.0f}")
        st.metric("📊 Harga Rata-rata", f"Rp {harga_rata:,.0f}")
    
    # Preview data
    st.subheader("👁️ Preview Data")
    st.dataframe(df_cleaned.tail(10), use_container_width=True)

# HALAMAN ANALISIS DATA
elif menu == "📈 Analisis Data":
    st.subheader("📈 Analisis Eksplorasi Data (EDA)")
    
    # Grafik utama
    st.subheader("📊 Trend Harga Beras Premium Papua")
    
    fig_line = px.line(
        df_cleaned, 
        x='Date', 
        y='Papua',
        title='Pergerakan Harga Beras Premium Papua dari Waktu ke Waktu',
        labels={'Papua': 'Harga (Rp)', 'Date': 'Tanggal'}
    )
    fig_line.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Harga (Rp)",
        hovermode='x unified'
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    # Analisis statistik
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribusi Harga")
        fig_hist = px.histogram(
            df_cleaned, 
            x='Papua',
            nbins=30,
            title='Distribusi Harga Beras Premium Papua',
            labels={'Papua': 'Harga (Rp)', 'count': 'Frekuensi'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("📈 Box Plot")
        fig_box = px.box(
            df_cleaned, 
            y='Papua',
            title='Sebaran Harga Beras Premium Papua',
            labels={'Papua': 'Harga (Rp)'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Statistik deskriptif
    st.subheader("📋 Statistik Deskriptif")
    stats_df = df_cleaned['Papua'].describe().round(2)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Rata-rata", f"Rp {stats_df['mean']:,.0f}")
        st.metric("📈 Maksimum", f"Rp {stats_df['max']:,.0f}")
    with col2:
        st.metric("📉 Minimum", f"Rp {stats_df['min']:,.0f}")
        st.metric("🎯 Median", f"Rp {stats_df['50%']:,.0f}")
    with col3:
        st.metric("📏 Std Deviasi", f"Rp {stats_df['std']:,.0f}")
        st.metric("📊 Q1", f"Rp {stats_df['25%']:,.0f}")
    with col4:
        st.metric("📊 Q3", f"Rp {stats_df['75%']:,.0f}")
        st.metric("📊 Range", f"Rp {stats_df['max'] - stats_df['min']:,.0f}")

# HALAMAN MODEL PREDIKSI
elif menu == "🤖 Model Prediksi":
    st.subheader("🤖 Model LSTM untuk Prediksi Harga")
    
    # Load model jika ada
    try:
        # Custom objects untuk mengatasi error deserialisasi
        custom_objects = {
            'mse': tf.keras.metrics.MeanSquaredError(),
            'keras.metrics.mse': tf.keras.metrics.MeanSquaredError()
        }
        
        model = load_model('lstm_model.h5', custom_objects=custom_objects)
        st.success("✅ Model LSTM berhasil dimuat!")
        
        # Informasi model
        st.subheader("ℹ️ Informasi Model")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Arsitektur Model:**
            - 🧠 LSTM Layer 1: 256 units
            - 🧠 LSTM Layer 2: 128 units  
            - 🧠 LSTM Layer 3: 64 units
            - 🔄 Dropout: 0.1 setiap layer
            - 🎯 Dense Layer: 32 units + output
            """)
        
        with col2:
            st.info("""
            **Parameter Training:**
            - 📊 Timesteps: 60 hari
            - 🔄 Epochs: 200 (dengan early stopping)
            - 📦 Batch size: 32
            - ⚡ Optimizer: Adam
            - 📉 Loss function: MSE
            """)
        
        # Proses data untuk prediksi
        data_normalized, scaler = normalize_data(df_cleaned.copy())
        
        # Buat sequences
        X, y = create_sequences(data_normalized)
        
        # Split data
        data_test_len = 30
        split_index = len(X) - data_test_len
        X_test = X[split_index:]
        y_test = y[split_index:]
        
        # Reshape untuk LSTM
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Prediksi
        if st.button("🚀 Jalankan Prediksi", type="primary"):
            with st.spinner("🔄 Sedang melakukan prediksi..."):
                y_pred = model.predict(X_test)
                
                # Denormalisasi
                y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                y_pred_rescaled = scaler.inverse_transform(y_pred).flatten()
                
                # Visualisasi hasil prediksi
                st.subheader("📈 Hasil Prediksi vs Data Aktual")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=y_test_rescaled,
                    mode='lines+markers',
                    name='Data Aktual',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))
                fig.add_trace(go.Scatter(
                    y=y_pred_rescaled,
                    mode='lines+markers',
                    name='Prediksi Model',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title='Perbandingan Prediksi LSTM vs Data Aktual',
                    xaxis_title='Hari ke-',
                    yaxis_title='Harga (Rp)',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Simpan hasil untuk evaluasi
                st.session_state.y_test = y_test_rescaled
                st.session_state.y_pred = y_pred_rescaled
                
                st.success("✅ Prediksi berhasil dilakukan!")
    
    except Exception as e:
        st.error(f"❌ Model tidak ditemukan: {e}")
        st.info("💡 Silakan jalankan notebook terlebih dahulu untuk melatih model.")

# HALAMAN EVALUASI MODEL
elif menu == "📋 Evaluasi Model":
    st.subheader("📋 Evaluasi Performa Model")
    
    if 'y_test' in st.session_state and 'y_pred' in st.session_state:
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        
        # Hitung metrik evaluasi
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Tampilkan metrik
        st.subheader("📊 Metrik Evaluasi")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="RMSE (Root Mean Square Error)",
                value=f"Rp {rmse:,.2f}",
                help="Semakin kecil semakin baik"
            )
        
        with col2:
            st.metric(
                label="MAE (Mean Absolute Error)",
                value=f"Rp {mae:,.2f}",
                help="Rata-rata kesalahan absolut"
            )
        
        with col3:
            st.metric(
                label="MAPE (Mean Absolute Percentage Error)",
                value=f"{mape:.2f}%",
                help="Kesalahan dalam bentuk persentase"
            )
        
        # Interpretasi hasil
        st.subheader("🎯 Interpretasi Hasil")
        
        if mape < 5:
            accuracy_level = "Sangat Baik"
            color = "green"
        elif mape < 10:
            accuracy_level = "Baik"
            color = "blue"
        elif mape < 20:
            accuracy_level = "Cukup"
            color = "orange"
        else:
            accuracy_level = "Perlu Perbaikan"
            color = "red"
        
        st.markdown(f"""
        <div style="padding: 1rem; border-left: 5px solid {color}; background-color: #f8f9fa;">
        <h4 style="color: {color};">Tingkat Akurasi: <span style="color: {color};">{accuracy_level}</span></h4>
        <p style="color: {color};">Model memiliki tingkat kesalahan rata-rata sebesar <strong>{mape:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Analisis residual
        st.subheader("📊 Analisis Residual")
        residuals = y_test - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_residual = px.scatter(
                x=y_pred, 
                y=residuals,
                title='Plot Residual vs Prediksi',
                labels={'x': 'Nilai Prediksi (Rp)', 'y': 'Residual (Rp)'}
            )
            fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residual, use_container_width=True)
        
        with col2:
            fig_hist_residual = px.histogram(
                x=residuals,
                nbins=20,
                title='Distribusi Residual',
                labels={'x': 'Residual (Rp)', 'y': 'Frekuensi'}
            )
            st.plotly_chart(fig_hist_residual, use_container_width=True)
        
        # Summary statistik residual
        st.subheader("📈 Statistik Residual")
        residual_stats = pd.Series(residuals).describe()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rata-rata Residual", f"{residual_stats['mean']:.2f}")
        with col2:
            st.metric("Std Deviasi Residual", f"{residual_stats['std']:.2f}")
        with col3:
            st.metric("Min Residual", f"{residual_stats['min']:.2f}")
        with col4:
            st.metric("Max Residual", f"{residual_stats['max']:.2f}")
    
    else:
        st.warning("⚠️ Belum ada hasil prediksi. Silakan jalankan prediksi terlebih dahulu di menu Model Prediksi.")
        if st.button("🔄 Kembali ke Model Prediksi"):
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🌾 Dashboard Prediksi Harga Beras Premium Papua | "
    "Dibuat dengan ❤️ menggunakan Streamlit"
    "</div>", 
    unsafe_allow_html=True
)