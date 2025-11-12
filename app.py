# app.py - NRL Predictor Web App (COMPLETE + 2026 MODE)
import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import joblib
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === GOOGLE VERIFICATION ===
if 'google' in st.query_params.get('file', []):
    st.title("NRL Predictor")
    st.info("App verified. Ready for NRL 2026!")
    st.stop()

# === GOOGLE SEARCH CONSOLE & ADSENSE ===
st.markdown("""
<meta name="google-site-verification" content="tBordOIFJNQRbb7Q7jalNy3A5WtqKmmeTbuf2R1Xh7Y" />
""", unsafe_allow_html=True)

st.markdown(
    '<meta name="google-adsense-account" content="ca-pub-2391186981906606">',
    unsafe_allow_html=True
)

st.markdown("""
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2391186981906606"
     crossorigin="anonymous"></script>
<ins class="adsbygoogle"
     style="display:block"
     data-ad-client="ca-pub-2391186981906606"
     data-ad-slot="3460661067"
     data-ad-format="auto"
     data-full-width-responsive="true"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({});
</script>
""", unsafe_allow_html=True)

# === CONFIG ===
DATA_FILE = "nrl_data.xlsx"
MODEL_FILE = "nrl_model.pkl"
LE_HOME_FILE = "le_home.pkl"
LE_AWAY_FILE = "le_away.pkl"

# === 1. DOWNLOAD DATA IF MISSING ===
if not os.path.exists(DATA_FILE):
    with st.spinner("Downloading NRL data from aussportsbetting.com..."):
        url = "https://www.aussportsbetting.com/historical_data/nrl.xlsx"
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(DATA_FILE, "wb") as f:
                f.write(r.content)
            st.success("NRL data downloaded!")
        else:
            st.error("Failed to download data. Using fallback.")
            DATA_FILE = None

# === 2. LOAD DATA (with fallback) ===
if DATA_FILE and os.path.exists(DATA_FILE):
    df = pd.read_excel(DATA_FILE, header=1)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Home Team', 'Away Team', 'Home Score', 'Away Score'], inplace=True)
    df['Home Win'] = (df['Home Score'] > df['Away Score']).astype(int)
else:
    st.warning("Using fallback data (limited).")
    df = pd.DataFrame({
        'Home Team': ['Penrith Panthers', 'Melbourne Storm'],
        'Away Team': ['Brisbane Broncos', 'Sydney Roosters'],
        'Home Score': [28, 30],
        'Away Score': [24, 22],
        'Home Win': [1, 1]
    })

# === 3. TRAIN OR LOAD MODEL & ENCODERS ===
@st.cache_resource
def load_or_train_model():
    if all(os.path.exists(f) for f in [MODEL_FILE, LE_HOME_FILE, LE_AWAY_FILE]):
        model = joblib.load(MODEL_FILE)
        le_home = joblib.load(LE_HOME_FILE)
        le_away = joblib.load(LE_AWAY_FILE)
        st.info("ML Model loaded from cache.")
    else:
        with st.spinner("Training ML model (first run only)..."):
            le_home = LabelEncoder().fit(df['Home Team'])
            le_away = LabelEncoder().fit(df['Away Team'])
            df['Home Enc'] = le_home.transform(df['Home Team'])
            df['Away Enc'] = le_away.transform(df['Away Team'])
           
            X = df[['Home Enc', 'Away Enc']]
            y = df['Home Win']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
           
            joblib.dump(model, MODEL_FILE)
            joblib.dump(le_home, LE_HOME_FILE)
            joblib.dump(le_away, LE_AWAY_FILE)
        st.success("ML Model trained & saved!")
    return model, le_home, le_away

model, le_home, le_away = load_or_train_model()

# === 4. PREDICT FUNCTION (ML + Monte Carlo) ===
def predict_match(home, away):
    try:
        h_enc = le_home.transform([home])[0]
        a_enc = le_away.transform([away])[0]
        ml_prob = model.predict_proba([[h_enc, a_enc]])[0][1]

        # Monte Carlo with normal distribution
        home_hist = df[df['Home Team'] == home]['Home Score']
        away_hist = df[df['Away Team'] == away]['Away Score']
        h_mean = home_hist.mean()
        h_std = home_hist.std() or 1
        a_mean = away_hist.mean()
        a_std = away_hist.std() or 1

        wins = draws = 0
        for _ in range(5000):
            hs = np.random.normal(h_mean, h_std)
            asa = np.random.normal(a_mean, a_std)
            if hs > asa:
                wins += 1
            elif abs(hs - asa) < 0.5:
                draws += 1
        total = 5000

        return {
            "ML Win %": ml_prob,
            "Sim Home Win %": wins/total,
            "Sim Away Win %": (total - wins - draws)/total,
            "Sim Draw %": draws/total,
            "Avg Home Score": h_mean,
            "Avg Away Score": a_mean
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# === 2026 ROSTER BOOSTS (FIXED) ===
if season == "2026" and use_roster_boosts:
    st.sidebar.subheader("Roster Impact (Elo Boost)")
    roster_boosts = {
        "Wests Tigers": st.sidebar.slider("Luai to Tigers", -200, 200, 100),
        "Dolphins": st.sidebar.slider("Cobbo to Dolphins", -200, 200, 80),
        "Newcastle Knights": st.sidebar.slider("Dylan Brown to Knights", -200, 200, 70),
        "Sydney Roosters": st.sidebar.slider("DCE to Roosters", -200, 200, 90),
        "South Sydney Rabbitohs": st.sidebar.slider("Fifita to Rabbitohs", -200, 200, 85),
        "Parramatta Eels": st.sidebar.slider("Pezet to Eels", -200, 200, 75),
        "Gold Coast Titans": st.sidebar.slider("Fifita leaves Titans", -200, 200, -60),
        "Melbourne Storm": st.sidebar.slider("Pezet leaves Storm", -200, 200, -70),
    }  # <--- THIS } WAS MISSING!
else:
    roster_boosts = {}

