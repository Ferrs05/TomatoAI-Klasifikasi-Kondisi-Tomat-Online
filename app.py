import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="TomatoAI - Deteksi Kematangan", page_icon="🍅", layout="wide")

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1a1a1a;
    }
    
    /* Animasi Background Utama */
    .stApp {
        background: linear-gradient(45deg, #9B2226, #EE9B00, #E9D8A6);
        background-size: 200% 200%;
        animation: gerak 5s ease infinite;
    }

    @keyframes gerak {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main { background-color: transparent; }
    
    /* Latar Belakang Teks Hero (Glassmorphism) */
    .hero-text-box {
        background: rgba(255, 255, 255, 0.85);
        padding: 40px;
        border-radius: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.6);
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #0d1b2a;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #4b5563;
        margin-top: 15px;
        line-height: 1.6;
    }
    .highlight { color: #e63946; }
    
    .custom-info {
        background-color: #f8fafc;
        color: #0f172a;
        padding: 15px 20px;
        border-radius: 12px;
        border-left: 5px solid #e63946;
        margin-top: 25px;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* --- Result Card (DIUBAH AGAR LEBIH BESAR) --- */
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 50px 40px; /* Padding dibesarkan */
        border-radius: 20px;
        border: 2px dashed #e2e8f0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .status-label {
        color: #64748b; 
        font-weight: 700; 
        font-size: 1.2rem; /* Diperbesar */
        letter-spacing: 2px;
        margin-bottom: 0;
    }
    .pred-text { 
        font-size: 4rem; /* FONT HASIL SANGAT BESAR */
        font-weight: 800; 
        margin: 10px 0 20px 0; 
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    .confidence-text {
        font-size: 1.3rem; /* Teks tingkat kepercayaan dibesarkan */
        color: #4b5563;
    }
    .confidence-text b {
        font-size: 1.6rem; /* Angka persen lebih besar */
        color: #1a1a1a;
    }
    
    /* --- History Card --- */
    .history-card {
        border: 1px solid #eee;
        border-radius: 15px;
        padding: 15px;
        background: rgba(253, 253, 253, 0.95);
        text-align: center;
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .history-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    /* --- Footer CSS --- */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.85); /* Semi-transparan agar menyatu */
        backdrop-filter: blur(10px); /* Efek kaca */
        text-align: center;
        padding: 15px 0;
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 600;
        border-top: 1px solid rgba(0,0,0,0.05);
        z-index: 999; /* Memastikan footer selalu berada di atas elemen lain */
    }
    
    /* Memberikan ruang kosong di paling bawah halaman agar konten tidak tertutup footer */
    .main .block-container {
        padding-bottom: 100px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. INISIALISASI SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 4. FUNGSI MACHINE LEARNING ---
@st.cache_resource
def load_trained_model():
    try:
        return tf.keras.models.load_model('models/best_tomat_cnn.keras')
    except Exception as e:
        return None

def process_and_predict(image, model):
    img = image.convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    classes = ['Belum Matang', 'Matang', 'Rusak']
    idx = np.argmax(predictions)
    return classes[idx], predictions[0][idx]

# --- 5. NAVBAR HORIZONTAL ---
selected = option_menu(
    menu_title=None, 
    options=["Beranda", "Klasifikasi", "Statistik", "Riwayat"], 
    icons=["house", "camera-fill", "bar-chart-fill", "clock-history"], 
    menu_icon="cast", 
    default_index=0, 
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent !important", "border": "none"},
        "icon": {"color": "#e63946", "font-size": "18px"}, 
        "nav-link": {
            "font-size": "16px", 
            "text-align": "center", 
            "margin":"0px 10px", 
            "background-color": "rgba(255, 255, 255, 0.9)", 
            "color": "#1a1a1a",
            "border-radius": "50px", 
            "padding": "10px"
        },
        "nav-link-selected": {
            "background-color": "#ffffff", 
            "color": "#e63946", 
            "font-weight": "800", 
            "border": "2px solid #e63946",
            "box-shadow": "0 4px 10px rgba(0,0,0,0.1)"
        },
    }
)

# --- 6. ROUTING HALAMAN ---

if selected == "Beranda":
    st.write("<br><br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1], gap="large")
    
    with c1:
        st.markdown("""
        <div class="hero-text-box">
            <h1 class="hero-title">Deteksi Kondisi <span class="highlight">Kematangan Tomat</span></h1>
            <p class="hero-subtitle">Sistem cerdas berbasis Convolutional Neural Network (CNN) untuk membantu menganalisis kualitas tomat secara instan dan akurat. Tingkatkan efisiensi pemilahan hasil panen Anda bersama TomatoAI.</p>
            <div class="custom-info">
                👆 Mulai deteksi dengan memilih menu <b>Klasifikasi</b> di navigasi atas.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        img_url = "https://t3.ftcdn.net/jpg/01/85/36/76/360_F_185367679_Me2IGPUlNgmA3xJdbewSlT0jIM9RqGx2.jpg"
        st.image(img_url)

elif selected == "Klasifikasi":
    st.markdown("### 🔍 Unggah dan Analisis")
    st.write("Silakan unggah foto tomat Anda. Pastikan gambar fokus pada satu buah tomat dengan pencahayaan yang baik.")
    
    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Preview Gambar")
            
    with col_b:
        st.markdown("#### Hasil Klasifikasi")
        if uploaded_file:
            model = load_trained_model()
            if model:
                with st.spinner('Menganalisis pola citra...'):
                    label, prob = process_and_predict(image, model)
                    confidence = prob * 100
                    
                    color = "#2e7d32" if label == 'Matang' else "#f59e0b" if label == 'Belum Matang' else "#dc2626"
                    
                    st.markdown(f"""
                        <div class="result-card">
                            <p class="status-label">STATUS TOMAT TERDETEKSI</p>
                            <div class="pred-text" style="color: {color};">{label}</div>
                            <p class="confidence-text">Tingkat Kepercayaan Model: <b>{confidence:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if not any(item.get('filename') == uploaded_file.name for item in st.session_state.history):
                        st.session_state.history.append({
                            "filename": uploaded_file.name,
                            "image": image,
                            "label": label,
                            "confidence": confidence,
                            "color": color
                        })
            else:
                st.error("Model tidak ditemukan! Pastikan file 'best_tomat_cnn.keras' berada di dalam folder 'models/'.")
        else:
            st.info("Sistem menunggu input gambar dari panel di sebelah kiri.")

elif selected == "Statistik":
    st.markdown("### 📈 Statistik Pemrosesan")
    
    if len(st.session_state.history) > 0:
        total = len(st.session_state.history)
        matang = sum(1 for item in st.session_state.history if item["label"] == "Matang")
        belum = sum(1 for item in st.session_state.history if item["label"] == "Belum Matang")
        rusak = sum(1 for item in st.session_state.history if item["label"] == "Rusak")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Diuji", total)
        m2.metric("Kondisi Matang", matang)
        m3.metric("Belum Matang", belum)
        m4.metric("Kondisi Rusak", rusak)
        
        st.write("---")
        st.markdown("#### Distribusi Kualitas Tomat")
        df = pd.DataFrame({
            "Kondisi": ["Matang", "Belum Matang", "Rusak"], 
            "Jumlah": [matang, belum, rusak]
        })
        st.bar_chart(df.set_index("Kondisi"), color="#e63946")
    else:
        st.warning("Belum ada data statistik. Silakan lakukan klasifikasi terlebih dahulu untuk memunculkan grafik.")

elif selected == "Riwayat":
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown("### 🕒 Riwayat Deteksi Sesi Ini")
    with col_btn:
        if st.button("🗑️ Bersihkan Riwayat", use_container_width=True):
            st.session_state.history = []
            st.rerun()
            
    if len(st.session_state.history) == 0:
        st.info("Riwayat kosong. Data gambar yang Anda uji akan tersimpan sementara di halaman ini.")
    else:
        cols = st.columns(3)
        for idx, item in enumerate(reversed(st.session_state.history)):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="history-card">
                    <h4 style="color: {item['color']}; margin-bottom: 5px;">{item['label']}</h4>
                    <p style="font-size: 13px; color: #666; margin-bottom: 10px;">Akurasi: {item['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.image(item["image"])

# --- 7. FOOTER ---
st.markdown("""
    <div class="footer">
        Copyright © 2026 Muhammad Ferry Saputra. All Rights Reserved.
    </div>
""", unsafe_allow_html=True)