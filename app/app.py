# ============================================================
# Used Car Price Prediction — Streamlit App v2
# app/app.py
# ============================================================

import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import os

st.set_page_config(
    page_title = "Used Car Price Predictor",
    page_icon  = "🚗",
    layout     = "wide",
    initial_sidebar_state = "collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #f4f6f8; }
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

    /* Clean white cards */
    .card {
        background: white;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* Section headings */
    .sec-title {
        font-size: 13px;
        font-weight: 600;
        color: #374151;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    /* Slider value pill — shown via st.metric */
    div[data-testid="stMetric"] {
        background: #f9fafb;
        border-radius: 8px;
        padding: 6px 10px;
        border: 1px solid #e5e7eb;
    }
    div[data-testid="stMetricValue"] {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #111827 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 11px !important;
        color: #6b7280 !important;
    }

    /* Price display */
    .price-card {
        background: linear-gradient(135deg, #f0fdf8 0%, #e6f7f1 100%);
        border: 1.5px solid #86efcd;
        border-radius: 14px;
        padding: 1.75rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .price-label { font-size: 13px; color: #4b7c6a; margin-bottom: 4px; }
    .price-value { font-size: 44px; font-weight: 700; color: #065f46; line-height:1.1; }
    .price-rupees { font-size: 13px; color: #6b7280; margin-top: 6px; }

    /* Range boxes */
    .range-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 8px;
        margin-bottom: 1rem;
    }
    .range-box {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
    }
    .rb-label { font-size: 11px; color: #9ca3af; margin-bottom: 3px; }
    .rb-val   { font-size: 16px; font-weight: 600; color: #1f2937; }
    .rb-val.green { color: #065f46; }

    /* Insight rows */
    .insight {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 9px 10px;
        background: #f9fafb;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #f3f4f6;
    }
    .i-icon {
        font-size: 17px;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .i-text { font-size: 13px; color: #4b5563; line-height: 1.55; }
    .i-text b { color: #111827; }

    /* Model leaderboard */
    .lb-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 0;
        border-bottom: 1px solid #f3f4f6;
        font-size: 13px;
        color: #374151;
    }
    .lb-row:last-child { border-bottom: none; }
    .pill {
        font-size: 10px;
        background: #d1fae5;
        color: #065f46;
        padding: 2px 8px;
        border-radius: 99px;
        font-weight: 700;
        margin-left: 6px;
    }

    /* Progress bars */
    .bar-wrap { margin-bottom: 10px; }
    .bar-meta {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: #6b7280;
        margin-bottom: 4px;
    }
    .bar-track {
        height: 6px;
        background: #e5e7eb;
        border-radius: 3px;
        overflow: hidden;
    }

    /* Predict button */
    .stButton > button {
        width: 100% !important;
        background: #065f46 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1rem !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
    }
    .stButton > button:hover { background: #047857 !important; }

    /* Selectbox and slider label fix */
    label[data-testid="stWidgetLabel"] p {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #374151 !important;
    }

    /* Slider thumb colour */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #065f46 !important;
    }

    div[data-testid="stHeader"] { display:none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────
@st.cache_resource
def load_model_package():
    base = os.path.join(os.path.dirname(__file__), '..', 'models')
    with open(os.path.join(base, 'champion_model.pkl'),  'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(base, 'scaler.pkl'),          'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(base, 'feature_columns.pkl'), 'rb') as f:
        features = pickle.load(f)
    with open(os.path.join(base, 'model_metadata.json'), 'r') as f:
        meta = json.load(f)
    return model, scaler, features, meta


def predict_price(present_price, kms_driven, fuel_type,
                  seller_type, transmission, owner, year,
                  model, scaler, feature_cols,
                  needs_scaling, current_year=2024):
    car_age           = current_year - year
    input_dict = {
        'Present_Price_log': np.log1p(present_price),
        'Kms_Driven_log':    np.log1p(kms_driven),
        'Car_Age':           car_age,
        'Owner':             owner,
        'Fuel_Diesel':       1 if fuel_type   == 'Diesel'    else 0,
        'Fuel_Petrol':       1 if fuel_type   == 'Petrol'    else 0,
        'Seller_Type_enc':   1 if seller_type == 'Dealer'    else 0,
        'Transmission_enc':  1 if transmission== 'Manual'    else 0,
    }
    input_df = pd.DataFrame([input_dict])[feature_cols]
    if needs_scaling:
        input_df = pd.DataFrame(
            scaler.transform(input_df), columns=feature_cols)
    return round(float(np.expm1(model.predict(input_df)[0])), 2)


try:
    model, scaler, feature_cols, meta = load_model_package()
    needs_scaling = meta['needs_scaling']
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()


# ── Car brand list ────────────────────────────────────────────
CAR_BRANDS = [
    "Select brand...",
    "Maruti Suzuki", "Hyundai", "Honda", "Toyota",
    "Tata", "Mahindra", "Ford", "Volkswagen",
    "Renault", "Nissan", "Skoda", "Kia",
    "MG", "Jeep", "BMW", "Mercedes-Benz",
    "Audi", "Chevrolet", "Datsun", "Fiat", "Other"
]

CAR_MODELS = {
    "Maruti Suzuki": ["Swift","Baleno","Alto","Wagon R","Ertiga",
                      "Dzire","Vitara Brezza","Celerio","Ignis","S-Cross"],
    "Hyundai":       ["i20","Creta","Verna","i10","Tucson",
                      "Venue","Santro","Elantra","Alcazar"],
    "Honda":         ["City","Amaze","Jazz","WR-V","CR-V","HR-V"],
    "Toyota":        ["Innova","Fortuner","Etios","Corolla",
                      "Yaris","Camry","Urban Cruiser"],
    "Tata":          ["Nexon","Harrier","Safari","Tiago",
                      "Tigor","Altroz","Punch"],
    "Mahindra":      ["Scorpio","XUV500","Bolero","XUV300",
                      "Thar","KUV100","Marazzo"],
    "Ford":          ["EcoSport","Endeavour","Figo","Aspire","Freestyle"],
    "Volkswagen":    ["Polo","Vento","Taigun","Tiguan","Virtus"],
    "Renault":       ["Kwid","Duster","Triber","Kiger","Captur"],
    "Nissan":        ["Magnite","Micra","Terrano","Kicks"],
    "Skoda":         ["Rapid","Octavia","Superb","Kushaq","Slavia"],
    "Kia":           ["Seltos","Sonet","Carnival","Carens"],
    "MG":            ["Hector","Astor","Gloster","ZS EV"],
    "Jeep":          ["Compass","Wrangler","Meridian","Grand Cherokee"],
    "BMW":           ["3 Series","5 Series","X1","X3","X5"],
    "Mercedes-Benz": ["C-Class","E-Class","GLA","GLC","S-Class"],
    "Audi":          ["A4","A6","Q3","Q5","Q7"],
    "Chevrolet":     ["Beat","Cruze","Enjoy","Sail","Trailblazer"],
    "Datsun":        ["redi-GO","GO","GO+"],
    "Fiat":          ["Punto","Linea","Avventura"],
    "Other":         ["Other model"],
}


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:white;border-radius:14px;border:1px solid #e5e7eb;
            padding:1rem 1.5rem;margin-bottom:1.25rem;
            display:flex;align-items:center;justify-content:space-between">
  <div style="display:flex;align-items:center;gap:12px">
    <div style="width:42px;height:42px;background:#d1fae5;border-radius:10px;
                display:flex;align-items:center;justify-content:center;font-size:22px">
      🚗
    </div>
    <div>
      <div style="font-size:17px;font-weight:700;color:#111827">
        Used Car Price Predictor
      </div>
      <div style="font-size:12px;color:#9ca3af">
        CarDekho dataset &nbsp;·&nbsp;
        Linear Regression champion &nbsp;·&nbsp;
        R² = 0.9745
      </div>
    </div>
  </div>
  <div style="font-size:12px;font-weight:600;background:#d1fae5;color:#065f46;
              padding:5px 14px;border-radius:99px">
    Live prediction
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════
left, right = st.columns([1.05, 1], gap="medium")


# ══════════════════════════════════════════════════════════════
# LEFT — INPUTS
# ══════════════════════════════════════════════════════════════
with left:

    # ── Car brand and model ───────────────────────────────────
    st.markdown('<div class="sec-title">🚘 &nbsp;Car identity</div>',
                unsafe_allow_html=True)

    b_col, m_col = st.columns(2)
    with b_col:
        selected_brand = st.selectbox(
            "Car brand",
            options = CAR_BRANDS,
            index   = 0,
            help    = "Select the manufacturer brand"
        )
    with m_col:
        if selected_brand and selected_brand != "Select brand...":
            model_options = CAR_MODELS.get(selected_brand, ["Other model"])
        else:
            model_options = ["Select brand first"]
        selected_model = st.selectbox(
            "Car model",
            options = model_options,
            index   = 0,
            help    = "Select the specific car model"
        )

    # Show selected car name
    if selected_brand and selected_brand != "Select brand...":
        st.markdown(
            f'<div style="font-size:13px;color:#065f46;font-weight:600;'
            f'background:#d1fae5;padding:6px 12px;border-radius:8px;'
            f'margin-bottom:8px">Selected: {selected_brand} {selected_model}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Numeric inputs with clean metric display ──────────────
    st.markdown('<div class="sec-title">📋 &nbsp;Car details</div>',
                unsafe_allow_html=True)

    # Present price
    p1, p2 = st.columns([3, 1])
    with p1:
        present_price = st.slider(
            "Present price — showroom (₹ Lakhs)",
            min_value=0.5, max_value=50.0,
            value=9.85, step=0.05
        )
    with p2:
        st.metric("Showroom", f"₹{present_price:.2f}L")

    # Kms driven
    k1, k2 = st.columns([3, 1])
    with k1:
        kms_driven = st.slider(
            "Kilometres driven",
            min_value=100, max_value=300000,
            value=35000, step=500
        )
    with k2:
        kms_display = (f"{kms_driven/1000:.0f}k km"
                       if kms_driven >= 1000
                       else f"{kms_driven} km")
        st.metric("Mileage", kms_display)

    # Year
    y1, y2 = st.columns([3, 1])
    with y1:
        year = st.slider(
            "Year of manufacture",
            min_value=2000, max_value=2024,
            value=2017, step=1
        )
    with y2:
        car_age = 2024 - year
        st.metric("Age", f"{car_age} yrs")

    st.markdown("---")

    # ── Categorical inputs ────────────────────────────────────
    st.markdown('<div class="sec-title">🔧 &nbsp;Specifications</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        fuel_type = st.selectbox(
            "Fuel type",
            ["Petrol", "Diesel", "CNG"],
            help="Type of fuel"
        )
    with c2:
        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"],
            help="Gear type"
        )
    with c3:
        seller_type = st.selectbox(
            "Seller type",
            ["Dealer", "Individual"],
            help="Who is selling"
        )

    st.markdown("---")

    # ── Owner ─────────────────────────────────────────────────
    st.markdown('<div class="sec-title">👤 &nbsp;Ownership history</div>',
                unsafe_allow_html=True)

    owner = st.select_slider(
        "Number of previous owners",
        options=[0, 1, 2, 3],
        value=0
    )

    owner_info = {
        0: ("🟢", "#d1fae5", "#065f46",
            "First owner — commands the highest resale value"),
        1: ("🟡", "#fef9c3", "#713f12",
            "One previous owner — minor price reduction"),
        2: ("🟠", "#ffedd5", "#7c2d12",
            "Two previous owners — moderate price reduction"),
        3: ("🔴", "#fee2e2", "#7f1d1d",
            "Three or more owners — notable price impact"),
    }
    icon, bg, fg, msg = owner_info[owner]
    st.markdown(
        f'<div style="font-size:13px;color:{fg};font-weight:500;'
        f'background:{bg};padding:7px 12px;border-radius:8px;'
        f'margin-top:6px">{icon} &nbsp;{msg}</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ── Predict button ────────────────────────────────────────
    predict_clicked = st.button(
        "🔮  Predict Selling Price",
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════
# RIGHT — RESULTS
# ══════════════════════════════════════════════════════════════
with right:

    predicted = predict_price(
        present_price, kms_driven, fuel_type,
        seller_type, transmission, owner, year,
        model, scaler, feature_cols, needs_scaling
    )

    rmse        = meta['performance']['test_rmse_lakhs']
    lower       = round(max(0.1, predicted - rmse), 2)
    upper       = round(predicted + rmse, 2)
    depreciation= round((1 - predicted / present_price) * 100, 1)
    depreciation= max(0, min(depreciation, 100))
    rupees      = int(predicted * 100_000)

    car_label = (f"{selected_brand} {selected_model}"
                 if selected_brand != "Select brand..."
                 else "Your car")

    # ── Price card ────────────────────────────────────────────
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">{car_label}</div>
        <div class="price-value">₹{predicted:.2f} Lakhs</div>
        <div class="price-rupees">≈ ₹{rupees:,}</div>
    </div>

    <div class="range-grid">
        <div class="range-box">
            <div class="rb-label">Lower bound</div>
            <div class="rb-val">₹{lower:.2f}L</div>
        </div>
        <div class="range-box">
            <div class="rb-label">Predicted</div>
            <div class="rb-val green">₹{predicted:.2f}L</div>
        </div>
        <div class="range-box">
            <div class="rb-label">Upper bound</div>
            <div class="rb-val">₹{upper:.2f}L</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Progress bars ─────────────────────────────────────────
    dep_color = ("#dc2626" if depreciation > 60
                 else "#f59e0b" if depreciation > 30
                 else "#10b981")

    st.markdown(f"""
    <div class="bar-wrap">
        <div class="bar-meta">
            <span>Depreciation from showroom price</span>
            <span><b>{depreciation}%</b></span>
        </div>
        <div class="bar-track">
            <div style="width:{depreciation}%;height:6px;
                        border-radius:3px;background:{dep_color}"></div>
        </div>
    </div>
    <div class="bar-wrap">
        <div class="bar-meta">
            <span>Model confidence (R²)</span>
            <span><b>97.45%</b></span>
        </div>
        <div class="bar-track">
            <div style="width:97.45%;height:6px;
                        border-radius:3px;background:#065f46"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Price insights ────────────────────────────────────────
    st.markdown('<div class="sec-title">💡 &nbsp;Price insights</div>',
                unsafe_allow_html=True)

    kms_per_year = kms_driven / max(car_age, 1)

    insights = [
        (
            "📅", "#d1fae5",
            f"<b>{car_age}-year-old car</b> — " +
            ("relatively new, low depreciation." if car_age <= 3
             else "moderate age, expected depreciation." if car_age <= 7
             else "older car, heavy depreciation applied.")
        ),
        (
            "🛣️", "#fef3c7",
            f"<b>{kms_driven:,} km driven</b> — " +
            (f"{kms_per_year:,.0f} km/yr — low mileage, price premium."
             if kms_per_year < 10000
             else f"{kms_per_year:,.0f} km/yr — average mileage, normal pricing."
             if kms_per_year < 20000
             else f"{kms_per_year:,.0f} km/yr — high mileage, price reduction.")
        ),
        (
            "⛽", "#ede9fe",
            {
                "Petrol":  "<b>Petrol</b> — most common fuel type, standard market pricing.",
                "Diesel":  "<b>Diesel</b> — commands a price premium over petrol.",
                "CNG":     "<b>CNG</b> — niche fuel type, smaller buyer pool.",
            }[fuel_type]
        ),
        (
            "⚙️", "#e0f2fe",
            {
                "Manual":    "<b>Manual transmission</b> — more common, standard pricing.",
                "Automatic": "<b>Automatic transmission</b> — premium over manual.",
            }[transmission]
        ),
        (
            "🏪", "#fce7f3",
            {
                "Dealer":     "<b>Dealer listing</b> — typically priced higher with warranty/service.",
                "Individual": "<b>Individual seller</b> — often priced lower, direct negotiation.",
            }[seller_type]
        ),
        (
            "👤", "#d1fae5",
            {
                0: "<b>First owner</b> — highest resale confidence.",
                1: "<b>One previous owner</b> — minor value reduction.",
                2: "<b>Two previous owners</b> — moderate value reduction.",
                3: "<b>Three+ owners</b> — notable price impact.",
            }[owner]
        ),
    ]

    for emoji, bg, text in insights:
        st.markdown(f"""
        <div class="insight">
            <div class="i-icon" style="background:{bg}">{emoji}</div>
            <div class="i-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Model leaderboard ─────────────────────────────────────
    st.markdown('<div class="sec-title">🏆 &nbsp;Model leaderboard</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="lb-row">
        <span>Linear Regression<span class="pill">champion</span></span>
        <span style="font-weight:700;color:#065f46">R² 0.9745</span>
    </div>
    <div class="lb-row">
        <span style="color:#6b7280">XGBoost</span>
        <span style="font-weight:600">R² 0.9424</span>
    </div>
    <div class="lb-row">
        <span style="color:#6b7280">Random Forest</span>
        <span style="font-weight:600">R² 0.9067</span>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:12px;color:#9ca3af;padding:0.25rem">
    Built with Streamlit &nbsp;·&nbsp; CarDekho Dataset &nbsp;·&nbsp;
    scikit-learn &nbsp;·&nbsp; End-to-End ML Project
</div>
""", unsafe_allow_html=True)