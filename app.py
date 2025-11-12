# app.py - NRL Predictor Web App (FIXED)
import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Serve verification file (add after imports)
if 'google' in st.experimental_get_query_params().to_dict().get('file', []):
    with open('google-verification.html', 'r') as f:
        st.markdown(f.read(), unsafe_allow_html=True)

# Google Search Console Verification (Remove after indexing)
st.markdown("""
<meta name="google-site-verification" google-site-verification=tBordOIFJNQRbb7Q7jalNy3A5WtqKmmeTbuf2R1Xh7Y />
""", unsafe_allow_html=True)

# --- GOOGLE ADSENSE ACCOUNT TAG (Required for AdSense to work) ---
st.markdown(
    '<meta name="google-adsense-account" content="ca-pub-2391186981906606">',
    unsafe_allow_html=True
)

# --- ADS ---
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

# --- CONFIG ---
DATA_FILE = "nrl_data.xlsx"
MODEL_FILE = "nrl_model.pkl"
LE_HOME_FILE = "le_home.pkl"
LE_AWAY_FILE = "le_away.pkl"

# --- 1. DOWNLOAD DATA IF MISSING ---
if not os.path.exists(DATA_FILE):
    with st.spinner("Downloading NRL data..."):
        url = "https://www.aussportsbetting.com/historical_data/nrl.xlsx"
        r = requests.get(url)
        with open(DATA_FILE, "wb") as f:
            f.write(r.content)
    st.success("Data downloaded!")

# --- 2. LOAD DATA ---
df = pd.read_excel(DATA_FILE, header=1)
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Home Team', 'Away Team', 'Home Score', 'Away Score'], inplace=True)
df['Home Win'] = (df['Home Score'] > df['Away Score']).astype(int)

# --- 3. TRAIN OR LOAD MODEL & ENCODERS ---
@st.cache_resource
def load_or_train_model():
    if all(os.path.exists(f) for f in [MODEL_FILE, LE_HOME_FILE, LE_AWAY_FILE]):
        model = joblib.load(MODEL_FILE)
        le_home = joblib.load(LE_HOME_FILE)
        le_away = joblib.load(LE_AWAY_FILE)
    else:
        with st.spinner("Training model (first run only)..."):
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
    return model, le_home, le_away

model, le_home, le_away = load_or_train_model()

# --- 4. PREDICT FUNCTION ---
def predict_match(home, away):
    try:
        h_enc = le_home.transform([home])[0]
        a_enc = le_away.transform([away])[0]
        ml_prob = model.predict_proba([[h_enc, a_enc]])[0][1]

        # Monte Carlo
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
        st.error(f"Error: {e}")
        return None

# --- 5. STREAMLIT UI ---
st.title("NRL Win Predictor")
st.write("Select any two teams to predict the match outcome!")

teams = sorted(le_home.classes_)
col1, col2 = st.columns(2)

with col1:
    home = st.selectbox("Home Team", teams, index=teams.index("Penrith Panthers") if "Penrith Panthers" in teams else 0)
with col2:
    away = st.selectbox("Away Team", teams, index=teams.index("Melbourne Storm") if "Melbourne Storm" in teams else 0)

if st.button("Predict Match"):
    with st.spinner("Simulating 5,000 games..."):
        result = predict_match(home, away)
    
    if result:
        st.success(f"**{home} vs {away}**")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("ML Model", f"{home} Win", f"{result['ML Win %']:.1%}")
        with c2:
            st.metric("Monte Carlo", f"{home} Win", f"{result['Sim Home Win %']:.1%}")
        
        st.write("**Full Simulation:**")
        st.json({
            "Home Win Chance": f"{result['Sim Home Win %']:.1%}",
            "Away Win Chance": f"{result['Sim Away Win %']:.1%}",
            "Draw Chance": f"{result['Sim Draw %']:.1%}",
            "Expected Score": f"{result['Avg Home Score']:.0f} – {result['Avg Away Score']:.0f}"
        })
    else:

        st.error("Could not predict. Check team names.")





