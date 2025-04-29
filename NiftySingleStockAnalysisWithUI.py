import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
from gnews import GNews
import logging
import traceback

# Define your users
USERS = {
    "kalyan": "Kalyan@2025@",
    "kishore": "Kishore$2025$",
    "chaitu": "Chaitu@2025$",
    "somy": "Somy@2025@",
    "guest": "Password!2025#"
}
# List of Nifty 200 stocks

nifty_200 = {
        "ABB India": "ABB.NS",
        "ACC": "ACC.NS",
        "APL Apollo Tubes": "APLAPOLLO.NS",
        "AU Small Finance Bank": "AUBANK.NS",
        "Adani Energy Solutions": "ADANIENSOL.NS",
        "Adani Enterprises": "ADANIENT.NS",
        "Adani Green Energy": "ADANIGREEN.NS",
        "Adani Ports and Special Economic Zone": "ADANIPORTS.NS",
        "Adani Power": "ADANIPOWER.NS", 
        "Adani Total Gas": "ATGL.NS",
        "Aditya Birla Capital": "ABCAPITAL.NS",
        "Aditya Birla Fashion and Retail": "ABFRL.NS",
        "Alkem Laboratories": "ALKEM.NS",
        "Ambuja Cements": "AMBUJACEM.NS",
        "Apollo Hospitals Enterprise": "APOLLOHOSP.NS",
        "Apollo Tyres": "APOLLOTYRE.NS",
        "Ashok Leyland": "ASHOKLEY.NS",
        "Asian Paints": "ASIANPAINT.NS",
        "Astral": "ASTRAL.NS",
        "Aurobindo Pharma": "AUROPHARMA.NS",
        "Avenue Supermarts": "DMART.NS",
        "Axis Bank": "AXISBANK.NS",
        "BSE": "BSE.NS",
        "Bajaj Auto": "BAJAJ-AUTO.NS",
        "Bajaj Finance": "BAJFINANCE.NS",
        "Bajaj Finserv": "BAJAJFINSV.NS",
        "Bajaj Holdings & Investment": "BAJAJHLDNG.NS",
        "Bajaj Housing Finance": "BAJAJHFL.NS",
        "Bandhan Bank": "BANDHANBNK.NS",
        "Bank of Baroda": "BANKBARODA.NS",
        "Bank of India": "BANKINDIA.NS",
        "Bank of Maharashtra": "MAHABANK.NS",
        "Bharat Dynamics": "BDL.NS",
        "Bharat Electronics": "BEL.NS",
        "Bharat Forge": "BHARATFORG.NS",
        "Bharat Heavy Electricals": "BHEL.NS",
        "Bharat Petroleum Corporation": "BPCL.NS",
        "Bharti Airtel": "BHARTIARTL.NS",
        "Bharti Hexacom": "BHARTIHEXA.NS",
        "Biocon": "BIOCON.NS",
        "Bosch": "BOSCHLTD.NS",
        "Britannia Industries": "BRITANNIA.NS",
        "CG Power and Industrial Solutions": "CGPOWER.NS",
        "Canara Bank": "CANBK.NS",
        "Cholamandalam Investment and Finance Company": "CHOLAFIN.NS",
        "Cipla": "CIPLA.NS",
        "Coal India": "COALINDIA.NS",
        "Cochin Shipyard": "COCHINSHIP.NS",
        "Coforge": "COFORGE.NS",
        "Colgate Palmolive (India)": "COLPAL.NS",
        "Container Corporation of India": "CONCOR.NS",
        "Cummins India": "CUMMINSIND.NS",
        "DLF": "DLF.NS",
        "Dabur India": "DABUR.NS",
        "Divi's Laboratories": "DIVISLAB.NS",
        "Dixon Technologies (India)": "DIXON.NS",
        "Dr. Reddy's Laboratories": "DRREDDY.NS",
        "Dummy Siemens": "DUMMYSIEMS.NS",
        "Eicher Motors": "EICHERMOT.NS",
        "Escorts Kubota": "ESCORTS.NS",
        "Eternal": "ETERNAL.NS",
        "Exide Industries": "EXIDEIND.NS",
        "FSN E-Commerce Ventures": "NYKAA.NS",
        "Federal Bank": "FEDERALBNK.NS",
        "GAIL (India)": "GAIL.NS",
        "GMR Airports": "GMRAIRPORT.NS",
        "Glenmark Pharmaceuticals": "GLENMARK.NS",
        "Godrej Consumer Products": "GODREJCP.NS",
        "Godrej Properties": "GODREJPROP.NS",
        "Grasim Industries": "GRASIM.NS",
        "HCL Technologies": "HCLTECH.NS",
        "HDFC Asset Management Company": "HDFCAMC.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "HDFC Life Insurance Company": "HDFCLIFE.NS",
        "Havells India": "HAVELLS.NS",
        "Hero MotoCorp": "HEROMOTOCO.NS",
        "Hindalco Industries": "HINDALCO.NS",
        "Hindustan Aeronautics": "HAL.NS",
        "Hindustan Petroleum Corporation": "HINDPETRO.NS",
        "Hindustan Unilever": "HINDUNILVR.NS",
        "Hindustan Zinc": "HINDZINC.NS",
        "Housing & Urban Development Corporation": "HUDCO.NS",
        "Hyundai Motor India": "HYUNDAI.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "ICICI Lombard General Insurance Company": "ICICIGI.NS",
        "ICICI Prudential Life Insurance Company": "ICICIPRULI.NS",
        "IDFC First Bank": "IDFCFIRSTB.NS",
        "Indiabulls Housing Finance Ltd": "IBULHSGFIN.NS",
        "Indigo Paints Ltd": "INDIGOPNTS.NS",
        "Indraprastha Gas Ltd": "IGL.NS",
        "Indus Towers Ltd": "INDUSTOWER.NS",
        "IndusInd Bank Ltd": "INDUSINDBK.NS",
        "Info Edge (India) Ltd": "NAUKRI.NS",
        "Infosys Ltd": "INFY.NS",
        "InterGlobe Aviation Ltd": "INDIGO.NS",
        "IRB Infrastructure Developers Ltd": "IRB.NS",
        "ITC Ltd": "ITC.NS",
        "Jindal Steel & Power Ltd": "JINDALSTEL.NS",
        "JSW Energy Ltd": "JSWENERGY.NS",
        "JSW Steel Ltd": "JSWSTEEL.NS",
        "Jubilant FoodWorks Ltd": "JUBLFOOD.NS",
        "Kajaria Ceramics Ltd": "KAJARIACER.NS",
        "Kalpataru Projects International Ltd": "KPIL.NS",
        "Kalyan Jewellers India Ltd": "KALYANKJIL.NS",
        "Karur Vysya Bank Ltd": "KARURVYSYA.NS",
        "KPI Green Energy Ltd": "KPIGREEN.NS",
        "Kotak Mahindra Bank Ltd": "KOTAKBANK.NS",
        "KPIT Technologies Ltd": "KPITTECH.NS",
        "KPR Mill Ltd": "KPRMILL.NS",
        "L&T Finance Holdings Ltd": "LTFH.NS",
        "L&T Technology Services Ltd": "LTTS.NS",
        "Larsen & Toubro Ltd": "LT.NS",
        "LIC Housing Finance Ltd": "LICHSGFIN.NS",
        "Life Insurance Corporation of India": "LICI.NS",
        "Lupin Ltd": "LUPIN.NS",
        "Mahanagar Gas Ltd": "MGL.NS",
        "Mahindra & Mahindra Financial Services Ltd": "M&MFIN.NS",
        "Mahindra & Mahindra Ltd": "M&M.NS",
        "Manappuram Finance Ltd": "MANAPPURAM.NS",
        "Marico Ltd": "MARICO.NS",
        "Maruti Suzuki India Ltd": "MARUTI.NS",
        "Max Financial Services Ltd": "MFSL.NS",
        "Max Healthcare Institute Ltd": "MAXHEALTH.NS",
        "Mphasis Ltd": "MPHASIS.NS",
        "MRF Ltd": "MRF.NS",
        "MTAR Technologies Ltd": "MTARTECH.NS",
        "Multi Commodity Exchange of India Ltd": "MCX.NS",
        "Natco Pharma Ltd": "NATCOPHARM.NS",
        "Nava Ltd": "NAVA.NS",
        "Navi Technologies Ltd": "NAVINTECH.NS",
        "Nestle India Ltd": "NESTLEIND.NS",
        "Netweb Technologies India Ltd": "NETWEB.NS",
        "NHPC Ltd": "NHPC.NS",
        "NMDC Ltd": "NMDC.NS",
        "NTPC Ltd": "NTPC.NS",
        "Oil & Natural Gas Corporation Ltd": "ONGC.NS",
        "One97 Communications Ltd": "PAYTM.NS",
        "Page Industries Ltd": "PAGEIND.NS",
        "PB Fintech Ltd": "POLICYBZR.NS",
        "Piramal Enterprises Ltd": "PEL.NS",
        "PNB Housing Finance Ltd": "PNBHOUSING.NS",
        "Power Finance Corporation Ltd": "PFC.NS",
        "Power Grid Corporation of India Ltd": "POWERGRID.NS",
        "Punjab National Bank": "PNB.NS",
        "PVR INOX Ltd": "PVRINOX.NS",
        "Rail Vikas Nigam Ltd": "RVNL.NS",
        "RBL Bank Ltd": "RBLBANK.NS",
        "REC Ltd": "RECLTD.NS",
        "Reliance Industries Ltd": "RELIANCE.NS",
        "Samvardhana Motherson International Ltd": "MOTHERSON.NS",
        "Sanofi India Ltd": "SANOFI.NS",
        "SBI Cards and Payment Services Ltd": "SBICARD.NS",
        "SBI Life Insurance Company Ltd": "SBILIFE.NS",
        "State Bank of India": "SBIN.NS",
        "Siemens Ltd": "SIEMENS.NS",
        "Sona BLW Precision Forgings Ltd": "SONACOMS.NS",
        "Steel Authority of India Ltd": "SAIL.NS",
        "Sun Pharmaceutical Industries Ltd": "SUNPHARMA.NS",
        "Tata Chemicals Ltd": "TATACHEM.NS",
        "Tata Consumer Products Ltd": "TATACONSUM.NS",
        "Tata Elxsi Ltd": "TATAELXSI.NS",
        "Tata Motors Ltd": "TATAMOTORS.NS",
        "Tata Power Co Ltd": "TATAPOWER.NS",
        "Tata Steel Ltd": "TATASTEEL.NS",
        "Tech Mahindra Ltd": "TECHM.NS",
        "Titan Company Ltd": "TITAN.NS",
        "Torrent Pharmaceuticals Ltd": "TORNTPHARM.NS",
        "Torrent Power Ltd": "TORNTPOWER.NS",
        "Trent Ltd": "TRENT.NS",
        "TVS Motor Company Ltd": "TVSMOTOR.NS",
        "UltraTech Cement Ltd": "ULTRACEMCO.NS",
        "Union Bank of India": "UNIONBANK.NS",
        "United Spirits Ltd": "MCDOWELL-N.NS",
        "Vedanta Ltd": "VEDL.NS",       
        "Voltas Ltd": "VOLTAS.NS",
        "Wipro Ltd": "WIPRO.NS",
        "Zee Entertainment Enterprises Ltd": "ZEEL.NS",
        "Zomato Ltd": "ZOMATO.NS",
        # Add more stocks as needed
    }

# This is the mapping you need for news search when starting with a ticker
ticker_to_company_name_map = {ticker: name for name, ticker in nifty_200.items()}

# Dictionary mapping pattern names to descriptions
PATTERN_DESCRIPTIONS = {
    "Standard Doji": "Indicates indecision in the market. Neither buyers nor sellers gained control.",
    "Shooting Star Doji": "A bearish reversal pattern that occurs after an uptrend, suggesting a potential top.",
    "Dragonfly Doji": "A bullish reversal pattern that occurs after a downtrend, suggesting a potential bottom.",
    "Long-Legged Doji": "Indicates strong indecision with significant price movement in both directions during the period.",
    "Doji-like": "Similar to a Doji, indicates indecision, but with a very small real body.",

    "Hammer": "A bullish reversal pattern occurring after a downtrend (Bullish Reversal Shape), suggests buying pressure came in after an initial sell-off.",
    "Hanging Man": "A bearish reversal pattern occurring after an uptrend (Bearish Reversal Shape), suggests selling pressure came in after an initial rally.",
    "Shooting Star": "A bearish reversal pattern occurring after an uptrend (Bearish Reversal Shape), suggests buying pressure failed and sellers took control.",
    "Inverted Hammer": "A bullish reversal pattern occurring after a downtrend (Bullish Reversal Shape), suggests buyers are testing resistance levels.",

    "Bullish Engulfing": "A major bullish reversal pattern where a large bullish candle completely engulfs the body of the previous bearish candle.",
    "Bearish Engulfing": "A major bearish reversal pattern where a large bearish candle completely engulfs the body of the previous bullish candle.",

    "Morning Star": "A bullish reversal pattern consisting of three candles, signaling a potential bottom.",
    "Evening Star": "A bearish reversal pattern consisting of three candles, signaling a potential top.",

    "Bullish Harami": "A bullish reversal pattern (often called a 'pregnant' pattern) where a small bullish candle is contained within the body of the previous bearish candle.",
    "Bearish Harami": "A bearish reversal pattern where a small bearish candle is contained within the body of the previous bullish candle.",

    "Bullish Marubozu": "A strong bullish candle with little to no shadows, indicating strong buying conviction.",
    "Bearish Marubozu": "A strong bearish candle with little to no shadows, indicating strong selling conviction.",

    "Spinning Top": "Indicates indecision, similar to a Doji but with a slightly larger body. Suggests potential price reversal.",

    "Piercing Pattern": "A bullish reversal pattern where a bullish candle closes more than halfway up the body of the preceding bearish candle.",
    "Dark Cloud Cover": "A bearish reversal pattern where a bearish candle closes more than halfway down the body of the preceding bullish candle."

    # Add descriptions for any other patterns you add
}

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        entered_username = st.session_state["username"]
        entered_password = st.session_state["password"]
        
        if entered_username in USERS and USERS[entered_username] == entered_password:
            st.session_state["password_correct"] = True
            st.session_state["authenticated"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
            st.error("❌ Invalid username or password")

    if "authenticated" not in st.session_state:
        # First run, show login form
        st.text_input("Username", key="username", placeholder="Enter your username")
        st.text_input("Password", type="password", key="password", placeholder="Enter your password")
        st.button("Login", on_click=password_entered)
        return "login"
    elif not st.session_state["password_correct"]:
        # Password not correct, show login form with error
        st.text_input("Username", key="username", placeholder="Enter your username")
        st.text_input("Password", type="password", key="password", placeholder="Enter your password")
        st.button("Login", on_click=password_entered)
        return "login"
    else:
        # Password correct
        return "dashboard"

def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
   
    try:
        # Moving Averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        df['RSI'] = calculate_rsi(df['Close'], 14)
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Stochastic Oscillator
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        
        df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Volume indicators
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean() 
               
        df['CMF'] = calculate_cmf(df, n=20)  # or n=21 if you prefer

        return df
    except Exception as e:
        st.error(f"Error in calculate_technical_indicators: {str(e)}")
        return None
    
def calculate_cmf(df, n=20):
    """Calculate Chaikin Money Flow (CMF)"""
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)  # Handle division by zero
    mf_volume = mf_multiplier * df['Volume']
    cmf = mf_volume.rolling(window=n).sum() / df['Volume'].rolling(window=n).sum()
    return cmf

def calculate_rsi(data, periods=14):
    """Calculate RSI using manual calculation"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def check_candlestick_patterns(df):
    """Check for advanced candlestick patterns"""
    try:
        # Ensure we have enough data for the patterns we want to check
        # Some patterns need 2, 3, or even more candles. Let's ensure we have at least 4 for safety.
        if len(df) < 4:
            return []

        # Get the last few days of data needed for pattern recognition
        # We'll look at the last 3 candles for most patterns, but access up to the 4th last if needed.
        last_candles = df.tail(4)

        # Check if we have all required columns in the relevant slice
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in last_candles.columns for col in required_columns):
            return []

        # Get the last few candles (most recent is candle_1)
        candle_4 = last_candles.iloc[0] # 4th last candle
        candle_3 = last_candles.iloc[1] # 3rd last candle
        candle_2 = last_candles.iloc[2] # 2nd last candle
        candle_1 = last_candles.iloc[3] # Last candle (most recent)


        patterns = []

        # Helper functions (already in your code, just keeping here for context)
        def is_bullish(candle):
            # Use .item() to get the scalar value from a potential Series element
            return candle['Close'].item() > candle['Open'].item()

        def is_bearish(candle):
            return candle['Close'].item() < candle['Open'].item()

        def get_candle_props(candle):
            c_open = candle['Open'].item()
            c_high = candle['High'].item()
            c_low = candle['Low'].item()
            c_close = candle['Close'].item()
            
            body = abs(c_close - c_open)
            upper_shadow = c_high - max(c_open, c_close)
            lower_shadow = min(c_open, c_close) - c_low
            total_range = c_high - c_low
            
            # Return values, ensuring they are non-negative in case of data anomalies
            return max(0, body), max(0, upper_shadow), max(0, lower_shadow), max(0, total_range)


        # Get candle properties for the last few candles
        body_1, upper_shadow_1, lower_shadow_1, total_range_1 = get_candle_props(candle_1)
        body_2, upper_shadow_2, lower_shadow_2, total_range_2 = get_candle_props(candle_2)
        body_3, upper_shadow_3, lower_shadow_3, total_range_3 = get_candle_props(candle_3)
        # body_4, upper_shadow_4, lower_shadow_4, total_range_4 = get_candle_props(candle_4) # Example if 4th candle props are needed

        # --- Existing Patterns (Refined slightly) ---

        # 1. Doji Pattern (Existing variations)
        if total_range_1 > 0 and body_1 <= 0.1 * total_range_1:
            # Added a check that body is not zero for the shadow checks to avoid division by zero logic implicitly
            if body_1 > 0 and upper_shadow_1 > 2 * body_1 and lower_shadow_1 <= body_1:
                patterns.append("Shooting Star Doji")
            elif body_1 > 0 and lower_shadow_1 > 2 * body_1 and upper_shadow_1 <= body_1:
                patterns.append("Dragonfly Doji")
            elif body_1 > 0 and upper_shadow_1 > body_1 and lower_shadow_1 > body_1:
                patterns.append("Long-Legged Doji")
            elif body_1 == 0: # Exactly zero body
                 patterns.append("Standard Doji")
            else: # Small body, not fitting specific Doji types
              patterns.append("Doji-like (Small Body)")


        # 2. Hammer / Hanging Man (Based on candle_1 shape and location)
        # Hammer is bullish, Hanging Man is bearish, but they have the same shape
        # Shape check: small upper shadow, small body, long lower shadow (> 2*body)
        if upper_shadow_1 <= body_1 and lower_shadow_1 > 2 * body_1 and body_1 > 0: # Ensure body > 0 to avoid division by zero issues with body comparison
            # Location check relative to recent price action (Simplified check using candle_2)
            # Hammer: occurs after a downtrend (candle_2 close < candle_2 open), candle_1 opens near low of candle_2
            if is_bearish(candle_2) and candle_1['Open'].item() <= candle_2['Close'].item(): # Simplified location check
                patterns.append("Hammer (Bullish Reversal Shape)")
            # Hanging Man: occurs after an uptrend (candle_2 close > candle_2 open), candle_1 opens near high of candle_2
            elif is_bullish(candle_2) and candle_1['Open'].item() >= candle_2['Close'].item(): # Simplified location check
                 patterns.append("Hanging Man (Bearish Reversal Shape)")


        # 3. Shooting Star (Based on candle_1 shape and location)
        # Shape check: small lower shadow, small body, long upper shadow (> 2*body)
        if lower_shadow_1 <= body_1 and upper_shadow_1 > 2 * body_1 and body_1 > 0: # Ensure body > 0
             # Location check relative to recent price action (Simplified check using candle_2)
             # Occurs after an uptrend (candle_2 close > candle_2 open), candle_1 opens near high of candle_2
            if is_bullish(candle_2) and candle_1['Open'].item() >= candle_2['Close'].item(): # Simplified location check
                patterns.append("Shooting Star (Bearish Reversal Shape)")


        # 4. Engulfing (Needs candle_1 and candle_2)
        # Bullish Engulfing: bearish candle_2, bullish candle_1, candle_1 body engulfs candle_2 body
        if (is_bearish(candle_2) and is_bullish(candle_1) and body_2 > 0 and body_1 > body_2 and # Ensure bodies > 0
            candle_1['Close'].item() >= candle_2['Open'].item() and # Candle 1 closes above Candle 2 open
            candle_1['Open'].item() <= candle_2['Close'].item()):    # Candle 1 opens below Candle 2 close
            patterns.append("Bullish Engulfing")
        # Bearish Engulfing: bullish candle_2, bearish candle_1, candle_1 body engulfs candle_2 body
        elif (is_bullish(candle_2) and is_bearish(candle_1) and body_2 > 0 and body_1 > body_2 and # Ensure bodies > 0
              candle_1['Open'].item() >= candle_2['Close'].item() and # Candle 1 opens above Candle 2 close
              candle_1['Close'].item() <= candle_2['Open'].item()):    # Candle 1 closes below Candle 2 open
            patterns.append("Bearish Engulfing")


        # 5. Morning Star (Needs candle_1, candle_2, candle_3)
        # Bearish candle_3, small body candle_2, bullish candle_1
        # Candle_2 gaps below candle_3 close, candle_1 closes well into candle_3 body
        if (is_bearish(candle_3) and body_3 > 0 and # Ensure body > 0
            abs(candle_2['Close'].item() - candle_2['Open'].item()) < body_3 * 0.5 and # Small body 2 relative to body 3
            candle_2['High'].item() < candle_3['Close'].item() and # Gap down (or near)
            is_bullish(candle_1) and body_1 > 0 and # Ensure body > 0
            candle_1['Open'].item() > candle_2['Close'].item() and # Gap up (or near)
            candle_1['Close'].item() > (candle_3['Open'].item() + candle_3['Close'].item()) / 2): # Candle 1 closes well into candle 3 body
            patterns.append("Morning Star")

        # 6. Evening Star (Needs candle_1, candle_2, candle_3)
        # Bullish candle_3, small body candle_2, bearish candle_1
        # Candle_2 gaps above candle_3 close, candle_1 closes well into candle_3 body
        elif (is_bullish(candle_3) and body_3 > 0 and # Ensure body > 0
              abs(candle_2['Close'].item() - candle_2['Open'].item()) < body_3 * 0.5 and # Small body 2 relative to body 3
              candle_2['Low'].item() > candle_3['Close'].item() and # Gap up (or near)
              is_bearish(candle_1) and body_1 > 0 and # Ensure body > 0
              candle_1['Open'].item() < candle_2['Close'].item() and # Gap down (or near)
              candle_1['Close'].item() < (candle_3['Open'].item() + candle_3['Close'].item()) / 2): # Candle 1 closes well into candle 3 body
            patterns.append("Evening Star")

        # 7. Harami (Needs candle_1 and candle_2)
        # Bullish Harami: Bearish candle_2, small bullish candle_1 completely within candle_2 body
        if (is_bearish(candle_2) and body_2 > 0 and # Ensure body > 0
            is_bullish(candle_1) and body_1 > 0 and body_1 < body_2 and # Ensure bodies > 0 and body 1 < body 2
            candle_1['High'].item() <= max(candle_2['Open'].item(), candle_2['Close'].item()) and # Candle 1 high within body 2
            candle_1['Low'].item() >= min(candle_2['Open'].item(), candle_2['Close'].item())):    # Candle 1 low within body 2
            patterns.append("Bullish Harami")
        # Bearish Harami: Bullish candle_2, small bearish candle_1 completely within candle_2 body
        elif (is_bullish(candle_2) and body_2 > 0 and # Ensure body > 0
              is_bearish(candle_1) and body_1 > 0 and body_1 < body_2 and # Ensure bodies > 0 and body 1 < body 2
              candle_1['High'].item() <= max(candle_2['Open'].item(), candle_2['Close'].item()) and # Candle 1 high within body 2
              candle_1['Low'].item() >= min(candle_2['Open'].item(), candle_2['Close'].item())):    # Candle 1 low within body 2
            patterns.append("Bearish Harami")


        # --- Added Patterns ---

        # 8. Marubozu (Needs candle_1)
        # Bullish Marubozu: Long bullish candle with almost no shadows
        if is_bullish(candle_1) and total_range_1 > 0 and body_1 >= 0.9 * total_range_1: # Body is at least 90% of total range
             patterns.append("Bullish Marubozu")
        # Bearish Marubozu: Long bearish candle with almost no shadows
        elif is_bearish(candle_1) and total_range_1 > 0 and body_1 >= 0.9 * total_range_1: # Body is at least 90% of total range
             patterns.append("Bearish Marubozu")


        # 9. Spinning Top (Needs candle_1)
        # Small body, upper shadow roughly equals lower shadow, both shadows relatively long compared to body
        if total_range_1 > 0 and body_1 > 0 and body_1 < 0.3 * total_range_1: # Small body (e.g., less than 30% of range) and body > 0
             # Shadows are relatively equal and longer than body
             if abs(upper_shadow_1 - lower_shadow_1) <= 0.5 * body_1 and upper_shadow_1 > body_1 and lower_shadow_1 > body_1:
                 patterns.append("Spinning Top")


        # 10. Inverted Hammer (Needs candle_1 shape and location)
        # Shape check: small lower shadow, small body, long upper shadow (> 2*body) - same shape as Shooting Star
        # Location check: occurs after a downtrend (candle_2 close < candle_2 open), candle_1 opens and closes near the *low* of the range
        if lower_shadow_1 <= body_1 and upper_shadow_1 > 2 * body_1 and body_1 > 0: # Ensure body > 0
             # Check if it's a potential Inverted Hammer (occurs after downtrend)
            if is_bearish(candle_2) and max(candle_1['Open'].item(), candle_1['Close'].item()) <= candle_1['Low'].item() + 0.3 * total_range_1: # Open/Close in lower part of range
                 patterns.append("Inverted Hammer (Bullish Reversal Shape)")


        # 11. Piercing Pattern (Needs candle_1 and candle_2)
        # Bullish reversal: Bearish candle_2, bullish candle_1 opens below candle_2 low, closes > 50% into candle_2 body
        if (is_bearish(candle_2) and body_2 > 0 and # Ensure body > 0
            is_bullish(candle_1) and body_1 > 0 and # Ensure body > 0
            candle_1['Open'].item() < candle_2['Low'].item() and # Gap down on open
            candle_1['Close'].item() > (candle_2['Open'].item() + candle_2['Close'].item()) / 2): # Closes more than halfway up candle 2 body
            patterns.append("Piercing Pattern")

        # 12. Dark Cloud Cover (Needs candle_1 and candle_2)
        # Bearish reversal: Bullish candle_2, bearish candle_1 opens above candle_2 high, closes > 50% into candle_2 body (from the top)
        elif (is_bullish(candle_2) and body_2 > 0 and # Ensure body > 0
              is_bearish(candle_1) and body_1 > 0 and # Ensure body > 0
              candle_1['Open'].item() > candle_2['High'].item() and # Gap up on open
              candle_1['Close'].item() < (candle_2['Open'].item() + candle_2['Close'].item()) / 2): # Closes more than halfway down candle 2 body
            patterns.append("Dark Cloud Cover")


        # Add more patterns here following similar logic...

        # Remove potential duplicates if a single candle fits multiple basic shapes (e.g., a perfect Doji is also a Spinning Top shape)
        # This is a simple way to prioritize more specific pattern names if overlaps occur.
        # You might want to refine the pattern logic to be mutually exclusive where possible.
        # For now, let's just remove pure "Doji-like" if a more specific Doji or Spinning Top is found.
        if "Doji-like (Small Body)" in patterns and (
            "Standard Doji" in patterns or
            "Shooting Star Doji" in patterns or
            "Dragonfly Doji" in patterns or
            "Long-Legged Doji" in patterns or
            "Spinning Top" in patterns):
             patterns.remove("Doji-like (Small Body)")

        # Clean up shape names if a more specific pattern is identified that implies the shape
        if "Hammer (Bullish Reversal Shape)" in patterns:
             if "Hammer" in patterns: patterns.remove("Hammer") # Remove generic if specific is found
             if "Hanging Man (Bearish Reversal Shape)" in patterns: patterns.remove("Hanging Man (Bearish Reversal Shape)") # These shapes are mutually exclusive based on trend check now

        if "Shooting Star (Bearish Reversal Shape)" in patterns:
              if "Shooting Star" in patterns: patterns.remove("Shooting Star") # Remove generic if specific is found

        if "Inverted Hammer (Bullish Reversal Shape)" in patterns:
              # No generic 'Inverted Hammer' in original code, but good to keep this structure
              pass # No action needed currently, but useful if a generic was added later


        return sorted(list(set(patterns))) # Return unique patterns, sorted alphabetically

    except Exception as e:
        # Use logging for errors in functions
        logging.error(f"Error in check_candlestick_patterns: {str(e)}")
        # Print traceback for debugging
        traceback.print_exc()
        return []

def calculate_target_and_sl(df):
    """Calculate target price and stop loss using ATR and price action"""
    try:
        # Calculate ATR
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        current_price = float(df['Close'].iloc[-1])
        current_atr = float(atr.iloc[-1])
        
        # Calculate resistance levels using recent highs
        recent_highs = df['High'].rolling(window=20).max()
        resistance_level = float(recent_highs.iloc[-1])
        
        # Calculate support levels using recent lows
        recent_lows = df['Low'].rolling(window=20).min()
        support_level = float(recent_lows.iloc[-1])
        
        # Calculate target price (resistance level or 2x ATR above current price)
        target_1 = current_price + (2 * current_atr)
        target_2 = resistance_level
        target_price = max(target_1, target_2)
        
        # Calculate stop loss (support level or 1x ATR below current price)
        sl_1 = current_price - current_atr
        sl_2 = support_level
        stop_loss = max(sl_1, sl_2)
        
        # Calculate risk-reward ratio
        risk = current_price - stop_loss
        reward = target_price - current_price
        risk_reward_ratio = reward / risk if risk != 0 else 0
        
        return {
            'target_price': round(target_price, 2),
            'stop_loss': round(stop_loss, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'atr': round(current_atr, 2)
        }
    except Exception as e:
        st.error(f"Error in calculate_target_and_sl: {str(e)}")
        return None

def check_candlestick_patterns(df):
    """Check for advanced candlestick patterns"""   
    try:
        # Ensure we have enough data
        if len(df) < 3:
            return []
            
        # Get the last 3 days of data
        last_3_days = df.tail(3)
        
        # Check if we have all required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in last_3_days.columns for col in required_columns):
            return []
        
        # Get the last 3 candles
        candle_3 = last_3_days.iloc[0]
        candle_2 = last_3_days.iloc[1]
        candle_1 = last_3_days.iloc[2]

        patterns = []

        def is_bullish(candle):
            return candle['Close'].item() > candle['Open'].item()

        def is_bearish(candle):
            return candle['Close'].item() < candle['Open'].item()

        def get_candle_props(candle):
            body = abs(candle['Close'].item() - candle['Open'].item())
            upper_shadow = candle['High'].item() - max(candle['Open'].item(), candle['Close'].item())
            lower_shadow = min(candle['Open'].item(), candle['Close'].item()) - candle['Low'].item()
            total_range = candle['High'].item() - candle['Low'].item()
            return body, upper_shadow, lower_shadow, total_range

        # Get candle properties
        body_1, upper_shadow_1, lower_shadow_1, total_range_1 = get_candle_props(candle_1)
        body_2, upper_shadow_2, lower_shadow_2, total_range_2 = get_candle_props(candle_2)
        body_3, upper_shadow_3, lower_shadow_3, total_range_3 = get_candle_props(candle_3)

        # 1. Doji Pattern
        if total_range_1 > 0 and body_1 <= 0.1 * total_range_1:
            if upper_shadow_1 > 2 * body_1 and lower_shadow_1 <= body_1:
                patterns.append("Shooting Star Doji")
            elif lower_shadow_1 > 2 * body_1 and upper_shadow_1 <= body_1:
                patterns.append("Dragonfly Doji")
            elif upper_shadow_1 > body_1 and lower_shadow_1 > body_1:
                patterns.append("Long-Legged Doji")
            else:
                patterns.append("Doji")

        # 2. Hammer / Hanging Man
        if lower_shadow_1 > 2 * body_1 and upper_shadow_1 <= body_1:
            if is_bullish(candle_1):
                patterns.append("Hammer")
            else:
                patterns.append("Hanging Man")

        # 3. Shooting Star
        if upper_shadow_1 > 2 * body_1 and lower_shadow_1 <= body_1:
            patterns.append("Shooting Star")

        # 4. Engulfing
        if (is_bearish(candle_2) and is_bullish(candle_1) and
            candle_1['Close'].item() > candle_2['Open'].item() and
            candle_1['Open'].item() < candle_2['Close'].item() and
            body_1 > body_2):
            patterns.append("Bullish Engulfing")
        if (is_bullish(candle_2) and is_bearish(candle_1) and
            candle_1['Open'].item() > candle_2['Close'].item() and
            candle_1['Close'].item() < candle_2['Open'].item() and
            body_1 > body_2):
            patterns.append("Bearish Engulfing")

        # 5. Morning Star
        if (is_bearish(candle_3) and
            abs(candle_2['Close'].item() - candle_2['Open'].item()) < body_1 * 0.3 and
            is_bullish(candle_1) and
            candle_1['Close'].item() > (candle_3['Open'].item() + candle_3['Close'].item()) / 2):
            patterns.append("Morning Star")

        # 6. Evening Star
        if (is_bullish(candle_3) and
            abs(candle_2['Close'].item() - candle_2['Open'].item()) < body_1 * 0.3 and
            is_bearish(candle_1) and
            candle_1['Close'].item() < (candle_3['Open'].item() + candle_3['Close'].item()) / 2):
            patterns.append("Evening Star")

        # 7. Harami
        if (is_bearish(candle_2) and is_bullish(candle_1) and
            candle_1['High'].item() <= candle_2['Open'].item() and
            candle_1['Low'].item() >= candle_2['Close'].item()):
            patterns.append("Bullish Harami")
        elif (is_bullish(candle_2) and is_bearish(candle_1) and
              candle_1['High'].item() <= candle_2['Close'].item() and
              candle_1['Low'].item() >= candle_2['Open'].item()):
            patterns.append("Bearish Harami")

        return patterns

    except Exception as e:
        print(f"Error in check_candlestick_patterns: {str(e)}")
        return []
    
def get_news_sentiment(ticker):
    """Get news sentiment for a given stock"""
    try:        
        company_name = ticker_to_company_name_map.get(ticker, ticker.split('.')[0])
       
        # Initialize Google News
        google_news = GNews(language='en', country='IN', period='7d', max_results=5)        
        # Search for news
        news_items = google_news.get_news(f"{company_name} stock NSE")
       
        if not news_items:
            return {
                "sentiment_score": 0,
                "sentiment_label": "Neutral",
                "latest_news": []
            }

        # Calculate sentiment for each news item
        sentiments = []
        latest_news = []
        
        for item in news_items:
            # Get sentiment using TextBlob
            blob = TextBlob(item['title'])
            sentiment = blob.sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)
            sentiments.append(sentiment)
           
            # Store news details
            latest_news.append({
                "title": item['title'],
                "date": item['published date'],
                "sourceTitle": item['publisher'].get('title', 'Unknown'),
                "sourceLink": item['publisher'].get('href', 'Unknown'),
                "sentiment": "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
            })
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Determine overall sentiment label
        if avg_sentiment > 0.1:
            sentiment_label = "Positive"
        elif avg_sentiment < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
             
        return {
            "sentiment_score": round(avg_sentiment, 2),
            "sentiment_label": sentiment_label,
            "latest_news": latest_news[:3]  # Return top 3 news items
        }
        
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {str(e)}")
        return {
            "sentiment_score": 0,
            "sentiment_label": "Neutral",
            "latest_news": []
        }   
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {str(e)}")
        return {
            "sentiment_score": 0,
            "sentiment_label": "Neutral",
            "latest_news": []
        }   

def calculate_support_levels(df):
    """Calculate support levels using pivot points and swing lows"""
    try:
        # Calculate pivot points
        df['Lowest'] = df['Low'].rolling(window=10).min()
        df['Highest'] = df['High'].rolling(window=10).max()
        
        # Find swing lows
        swing_lows = []
        for i in range(2, len(df) - 2):
            if df['Low'].iloc[i] <= df['Low'].iloc[i-2:i].min() and \
               df['Low'].iloc[i] <= df['Low'].iloc[i+1:i+3].min():
                swing_lows.append(float(df['Low'].iloc[i]))  # Convert to float
        
        # Calculate support levels
        support_levels = []
        if swing_lows:
            # Sort and get unique levels
            swing_lows = sorted(set(swing_lows))
            # Get the most recent significant levels
            support_levels = swing_lows[-3:]  # Get last 3 significant levels
            support_levels.reverse()  # Put them in ascending order
        
        # Add recent pivot points
        if len(df) > 10:
            pivot = float((df['Lowest'].iloc[-1] + df['Close'].iloc[-1] * 2 + df['Highest'].iloc[-1]) / 4)
            support_levels.append(pivot)
        
        # Remove duplicates and sort
        support_levels = sorted(set(support_levels))  # Sort in ascending order
        
        # Round to 2 decimal places and return clean float values
        return [round(float(x), 2) for x in support_levels[:3]]  # Return top 3 support levels rounded to 2 decimals
    except Exception as e:
        print(f"Error calculating support levels: {str(e)}")
        return [round(float(df['Low'].iloc[-1] * 0.99), 2), 
                round(float(df['Low'].iloc[-1] * 0.98), 2), 
                round(float(df['Low'].iloc[-1] * 0.97), 2)]  # Fallback levels rounded to 2 decimals

def calculate_resistance_levels(df):
    """Calculate resistance levels using pivot points and swing highs"""
    try:
        # Calculate pivot points
        df['Lowest'] = df['Low'].rolling(window=10).min()
        df['Highest'] = df['High'].rolling(window=10).max()
        
        # Find swing highs
        swing_highs = []
        for i in range(2, len(df) - 2):
            if df['High'].iloc[i] >= df['High'].iloc[i-2:i].max() and \
               df['High'].iloc[i] >= df['High'].iloc[i+1:i+3].max():
                swing_highs.append(float(df['High'].iloc[i]))  # Convert to float
        
        # Calculate resistance levels
        resistance_levels = []
        if swing_highs:
            # Sort and get unique levels
            swing_highs = sorted(set(swing_highs))
            # Get the most recent significant levels
            resistance_levels = swing_highs[-3:]  # Get last 3 significant levels
            resistance_levels.reverse()  # Put them in ascending order
        
        # Add recent pivot points
        if len(df) > 10:
            pivot = float((df['Lowest'].iloc[-1] + df['Close'].iloc[-1] * 2 + df['Highest'].iloc[-1]) / 4)
            resistance_levels.append(pivot)
        
        # Remove duplicates and sort
        resistance_levels = sorted(set(resistance_levels), reverse=True)  # Sort in descending order
        
        # Round to 2 decimal places and return clean float values
        return [round(float(x), 2) for x in resistance_levels[:3]]  # Return top 3 resistance levels rounded to 2 decimals
    except Exception as e:
        print(f"Error calculating resistance levels: {str(e)}")
        return [round(float(df['High'].iloc[-1] * 1.01), 2), 
                round(float(df['High'].iloc[-1] * 1.02), 2), 
                round(float(df['High'].iloc[-1] * 1.03), 2)]  # Fallback levels rounded to 2 decimals

def generate_analysis_summary(conditions, patterns, news_data, raw_score, strength_score):
    """Generate a comprehensive analysis summary with rich UI formatting."""
    summary_points_formatted = [] # Use a list to build formatted points

    # --- Trend Analysis ---
    trend_summary = []
    trend_color = 'neutral'
    if conditions.get("Strong Uptrend"):
        trend_summary.append("Strong Uptrend across multiple timeframes.")
        trend_color = 'positive'
    elif conditions.get("Price > MA20") and conditions.get("Price > MA50") and conditions.get("Price > MA200"):
         trend_summary.append("Price trending above key Moving Averages (MA20, MA50, MA200), indicating a Bullish trend across multiple timeframes.")
         trend_color = 'positive'
    elif conditions.get("Price > MA50"):
        trend_summary.append("Stock is in a Bullish trend on the medium-term (above MA50).")
        trend_color = 'positive'
    elif conditions.get("Price < MA200"):
        trend_summary.append("Stock is in a Bearish trend on the long-term (below MA200).")
        trend_color = 'negative'
    elif conditions.get("Price < MA20") and conditions.get("Price < MA50") and conditions.get("Price < MA200"):
         trend_summary.append("Price trending below key Moving Averages (MA20, MA50, MA200), indicating a Bearish trend across multiple timeframes.")
         trend_color = 'negative'
    else:
        trend_summary.append("Trend analysis shows Mixed or Neutral conditions.")
        trend_color = 'neutral'

    if trend_summary:
        # Format trend summary with color
        formatted_trend = f"<span class='{trend_color}'>{' '.join(trend_summary)}</span>"
        summary_points_formatted.append(f"**Trend Analysis:** {formatted_trend}")


    # --- Momentum Analysis ---
    momentum_summary = []
    momentum_color = 'neutral'
    # Check if conditions exist and are True
    if conditions.get("RSI Bullish") and conditions.get("MACD Bullish") and conditions.get("Stochastic Bullish"):
         momentum_summary.append("Momentum indicators (RSI, MACD, Stochastic) all showing Bullish signals.")
         momentum_color = 'positive'
    elif conditions.get("RSI Bullish") or conditions.get("MACD Bullish") or conditions.get("Stochastic Bullish"):
         bullish_indicators = [
             "RSI" if conditions.get("RSI Bullish") else None,
             "MACD" if conditions.get("MACD Bullish") else None,
             "Stochastic" if conditions.get("Stochastic Bullish") else None
         ]
         bullish_indicators = [ind for ind in bullish_indicators if ind is not None]
         momentum_summary.append(f"Some momentum indicators ({', '.join(bullish_indicators)}) show Bullish momentum.")
         momentum_color = 'positive' # Still leans positive if some are bullish

    elif conditions.get("RSI Overbought") or conditions.get("Stochastic Overbought"):
         overbought_indicators = [
             "RSI" if conditions.get("RSI Overbought") else None,
             "Stochastic" if conditions.get("Stochastic Overbought") else None
         ]
         overbought_indicators = [ind for ind in overbought_indicators if ind is not None]
         momentum_summary.append(f"Momentum indicators ({', '.join(overbought_indicators)}) showing Overbought conditions.")
         momentum_color = 'negative' # Overbought is often seen as bearish signal for entry

    elif conditions.get("RSI Oversold") or conditions.get("Stochastic Oversold"):
         oversold_indicators = [
             "RSI" if conditions.get("RSI Oversold") else None,
             "Stochastic" if conditions.get("Stochastic Oversold") else None
         ]
         oversold_indicators = [ind for ind in oversold_indicators if ind is not None]
         momentum_summary.append(f"Momentum indicators ({', '.join(oversold_indicators)}) showing Oversold conditions.")
         momentum_color = 'positive' # Oversold is often seen as bullish signal for entry

    elif conditions.get("MACD Bearish"):
         momentum_summary.append("MACD shows Bearish crossover or is negative.")
         momentum_color = 'negative'

    if not momentum_summary:
        momentum_summary.append("Momentum signals are mixed or neutral.")
        momentum_color = 'neutral'

    if momentum_summary:
        # Format momentum summary with color
        formatted_momentum = f"<span class='{momentum_color}'>{' '.join(momentum_summary)}</span>"
        summary_points_formatted.append(f"**Momentum Analysis:** {formatted_momentum}")


    # --- Pattern Analysis ---
    pattern_summary = []
    pattern_color = 'neutral'

    # Re-use the logic from the check_candlestick_patterns to classify found patterns
    bullish_patterns_found = [p for p in patterns if p in [
        "Hammer (Bullish Reversal Shape)", "Morning Star", "Bullish Engulfing", "Bullish Harami", "Dragonfly Doji",
        "Bullish Marubozu", "Inverted Hammer (Bullish Reversal Shape)", "Piercing Pattern"
        # Add other bullish patterns here
    ]]

    bearish_patterns_found = [p for p in patterns if p in [
        "Hanging Man (Bearish Reversal Shape)", "Evening Star", "Bearish Engulfing", "Bearish Harami", "Shooting Star (Bearish Reversal Shape)",
        "Shooting Star Doji", "Bearish Marubozu", "Dark Cloud Cover"
        # Add other bearish patterns here
    ]]

    neutral_patterns_found = [p for p in patterns if p not in bullish_patterns_found + bearish_patterns_found]


    if bullish_patterns_found and not bearish_patterns_found:
        pattern_summary.append(f"Potential bullish candlestick patterns detected: {', '.join(bullish_patterns_found)}.")
        pattern_color = 'positive'
    elif bearish_patterns_found and not bullish_patterns_found:
        pattern_summary.append(f"Potential bearish candlestick patterns detected: {', '.join(bearish_patterns_found)}.")
        pattern_color = 'negative'
    elif bullish_patterns_found and bearish_patterns_found:
        pattern_summary.append(f"Conflicting candlestick patterns detected. Bullish: {', '.join(bullish_patterns_found)}. Bearish: {', '.join(bearish_patterns_found)}.")
        pattern_color = 'neutral' # Conflicting is neutral/caution
    elif neutral_patterns_found:
        pattern_summary.append(f"Neutral or indecision patterns detected: {', '.join(neutral_patterns_found)}.")
        pattern_color = 'neutral'
    else:
        pattern_summary.append("No significant candlestick patterns detected recently.")
        pattern_color = 'neutral'

    if pattern_summary:
         # Format pattern summary with color
        formatted_pattern = f"<span class='{pattern_color}'>{' '.join(pattern_summary)}</span>"
        summary_points_formatted.append(f"**Pattern Analysis:** {formatted_pattern}")


    # --- Sentiment Analysis ---
    sentiment_summary = []
    sentiment_color = 'neutral'
    sentiment_label = news_data.get("sentiment_label", "Neutral")
    sentiment_score_val = news_data.get("sentiment_score", 'N/A')

    if sentiment_label == "Positive":
        sentiment_summary.append(f"News sentiment is Positive (Score: {sentiment_score_val:.2f}).")
        sentiment_color = 'positive'
    elif sentiment_label == "Negative":
        sentiment_summary.append(f"News sentiment is Negative (Score: {sentiment_score_val:.2f}).")
        sentiment_color = 'negative'
    else:
        sentiment_summary.append(f"News sentiment is Neutral (Score: {sentiment_score_val:.2f}).")
        sentiment_color = 'neutral'

    if sentiment_summary:
        # Format sentiment summary with color
        formatted_sentiment = f"<span class='{sentiment_color}'>{' '.join(sentiment_summary)}</span>"
        summary_points_formatted.append(f"**Sentiment Analysis:** {formatted_sentiment}")


    # --- Risk Management & Levels ---
    risk_levels_summary = []
    risk_color = 'neutral'

    if conditions.get("Good Risk-Reward"):
        risk_levels_summary.append("Calculated Risk-Reward Ratio is favorable.")
        risk_color = 'positive' # Favorable R:R is a positive risk factor
    else:
         risk_levels_summary.append("Calculated Risk-Reward Ratio may not be favorable.")
         # Color can remain neutral or be negative depending on how unfavorable

    if conditions.get("Reasonable Volatility"):
        risk_levels_summary.append("Volatility levels are within a reasonable range.")
        # Color can be neutral or slightly positive depending on view
    elif news_data.get("volatility") is not None:
         if news_data["volatility"] > 40:
              risk_levels_summary.append("Volatility is currently High.")
              # Color can be neutral or negative depending on view (high vol is higher risk)
         else:
              risk_levels_summary.append("Volatility is currently Low.")
              # Color can be neutral or positive depending on view (low vol might mean cheaper options but less movement)


    if conditions.get("Above Support"):
         risk_levels_summary.append("Price trading above a recent calculated support level.")
         risk_color = 'positive' # Above support is generally bullish/positive risk

    if conditions.get("Below Resistance"): # Add this condition check if you create it later
         risk_levels_summary.append("Price trading below a recent calculated resistance level.")
         # Color can remain neutral or be negative depending on view (below resistance is generally bearish/negative)


    if not risk_levels_summary:
        risk_levels_summary.append("Risk factors and price levels not fully assessed.")
        risk_color = 'neutral'

    if risk_levels_summary:
        # Decide the overall color for the Risk & Levels section
        # This is a bit tricky as points can be mixed. You could set color based on most dominant signal
        # For simplicity, let's just make the section title bold for now, or color based on the R:R primarily.
        if 'positive' in [risk_color]: # Check if any specific point made it positive
             formatted_risk = f"<span class='positive'>{' '.join(risk_levels_summary)}</span>"
             summary_points_formatted.append(f"**Risk & Levels:** {formatted_risk}")
        # Add other color checks here based on your preference
        else:
             formatted_risk = f"{' '.join(risk_levels_summary)}" # No overall color class applied
             summary_points_formatted.append(f"**Risk & Levels:** {formatted_risk}")



    # --- Overall Conclusion ---
    overall_conclusion = []
    conclusion_color = 'neutral'
    # You can add logic here to provide a more direct conclusion
    # based on the combination of factors, linking different points.
    # For example:
    if any("Strong Uptrend" in s or "Bullish trend across" in s for s in trend_summary) and (any("Bullish momentum" in s for s in momentum_summary) or any(p in bullish_patterns_found for p in patterns)):
        overall_conclusion.append("Overall technical signals suggest a potential continuation of bullish movement or a bullish reversal opportunity.")
        conclusion_color = 'positive'
    elif any("Bearish trend across" in s for s in trend_summary) and (any("Bearish momentum" in s for s in momentum_summary) or any(p in bearish_patterns_found for p in patterns)):
         overall_conclusion.append("Overall technical signals suggest a potential continuation of bearish movement or a bearish reversal risk.")
         conclusion_color = 'negative'
    elif any("Overbought conditions" in s for s in momentum_summary):
         overall_conclusion.append("Technical indicators are showing overbought conditions, suggesting caution or potential pullback.")
         conclusion_color = 'negative'
    elif any("Oversold conditions" in s for s in momentum_summary):
         overall_conclusion.append("Technical indicators are showing oversold conditions, suggesting potential for a bounce.")
         conclusion_color = 'positive'
    else:
         overall_conclusion.append("The analysis presents mixed signals, suggesting a neutral stance or waiting for clearer market direction.")
         conclusion_color = 'neutral'

    if overall_conclusion:
        # Format conclusion with color
        formatted_conclusion = f"<span class='{conclusion_color}'>{' '.join(overall_conclusion)}</span>"
        summary_points_formatted.append(f"**Conclusion:** {formatted_conclusion}")


    # --- Scoring Explanation ---
    scoring_explanation = (
        "**Scoring Explanation:** A dynamic strength score (0-100) is calculated based on a point system. "
        "Points are assigned based on factors like: Price position relative to Moving Averages, Momentum indicator values (RSI, MACD, Stochastic), "
        "Volume analysis, detected Candlestick Patterns, calculated Risk-Reward Ratio, Volatility levels, and News Sentiment. "
        "Positive indicators add points, while negative indicators subtract points. "
        "The total raw score is then normalized to a 0-100 scale, where a higher score indicates stronger bullish signals."
    )
    # This section is informational, keeping it neutral color is usually best
    summary_points_formatted.append(scoring_explanation)


    # Join all formatted points into the final summary string, separated by markdown list items
    return "\n".join([f"* {point}" for point in summary_points_formatted])
   
def generate_trade_recommendation(strength_score, conditions, raw_score):
    """Generate a trade recommendation based on the revised strength score and conditions."""
    # Prioritize conflicting or bearish signals
    if conditions["Bearish Pattern"]:
        return "Neutral / Avoid (Bearish pattern detected)"
    if conditions["Bullish Pattern"] and conditions["Bearish Pattern"]: # Explicitly check for conflict
         return "Neutral / Avoid (Conflicting patterns)"
    # Add other checks for strong bearish indicators if needed
    # Example: If RSI is below 30 (oversold usually, but context matters) AND trend is down...

    # Use the new strength score for recommendations if no strong bearish signal
    if strength_score >= 70:
        return "Strong Buy"
    elif strength_score >= 50:
        return "Buy"
    elif strength_score >= 30:
        return "Neutral / Hold"
    else:
        return "Neutral / Weak" # Or "Avoid"
 
def get_nifty_200_stocks():
    """Return hardcoded list of Nifty 200 stocks"""   
    print(nifty_200.__len__())
    # Sort alphabetically by company name
    sorted_stocks = dict(sorted(nifty_200.items()))
    return sorted_stocks

def analyze_stock(ticker):
    """Analyze a single stock with comprehensive technical and fundamental analysis"""
    try:
        print(f"Analyzing {ticker}...")

        # --- Download data ---
        print("Downloading data...")
        # Increased period for better MAs and ATR calculation
        df = yf.download(ticker, period="1y", interval="1d")
        if df.empty or len(df) < 200: # Ensure minimum data for MA200
            print(f"Error: Not enough historical data found for {ticker} for meaningful analysis (requires at least 200 days).")
            return None

        # --- Calculate Technical Indicators ---
        print("Calculating technical indicators...")
        df = calculate_technical_indicators(df)
        # Drop rows with NaN values resulting from indicator calculations
        df.dropna(inplace=True)
        if df.empty:
            print("Error: No valid data after calculating technical indicators.")
            return None

        # --- Calculate Target/SL/ATR ---
        print("Calculating target and stop loss...")
        price_levels = calculate_target_and_sl(df)
        if price_levels is None:
            print("Error: Failed to calculate target and stop loss.")
            return None        
       

        # --- Add ranges to the result dictionary ---
        # Add these to the 'result' dictionary just before the return statement
        # Format them as strings for display

        # --- Get Latest Values ---
        print("Getting latest values...")
         # Ensure df is not empty after dropna
        if df.empty:
             print("Error: DataFrame is empty after dropna.")
             return None

        latest = df.iloc[-1]

        # Safely access latest values, providing defaults or checking for NaN later
        current_price = float(latest.get('Close', np.nan))
        cmf = float(latest.get('CMF', np.nan))
         # --- Calculate Entry and Exit Ranges ---
        entry_range_lower = None
        entry_range_upper = None
        exit_range_lower = None
        exit_range_upper = None
        
        ma20 = float(latest.get('MA20', np.nan))
        ma50 = float(latest.get('MA50', np.nan))
        ma200 = float(latest.get('MA200', np.nan))
        rsi = float(latest.get('RSI', np.nan))
        macd = float(latest.get('MACD', np.nan))
        macd_signal = float(latest.get('MACD_signal', np.nan))
        stoch_k = float(latest.get('Stoch_K', np.nan))
        stoch_d = float(latest.get('Stoch_D', np.nan))
        volume = float(latest.get('Volume', np.nan))
        volume_ma20 = float(latest.get('Volume_MA20', np.nan))
        current_atr = price_levels.get('atr', np.nan)
        risk_reward_ratio = price_levels.get('risk_reward_ratio', 0) # Default R:R to 0 if not calculated
        cmf = float(latest.get('CMF', np.nan))

        # Ensure we have previous day's data for change calculations
        daily_change, volume_change = 0, 0
        if len(df) > 1:
            prev_close = float(df['Close'].iloc[-2])
            prev_volume = float(df['Volume'].iloc[-2])
            daily_change = ((current_price - prev_close) / prev_close) * 100 if prev_close != 0 and not np.isnan(current_price) and not np.isnan(prev_close) else 0
            volume_change = ((volume - prev_volume) / prev_volume) * 100 if prev_volume != 0 and not np.isnan(volume) and not np.isnan(prev_volume) else 0


        # --- Calculate Trends, Volatility, Levels ---
        print("Calculating trend directions...")
        # Use NaN checks as MA values might be NaN for initial rows
        short_term_trend = "Bullish" if not np.isnan(current_price) and not np.isnan(ma20) and current_price > ma20 else "Bearish" if not np.isnan(current_price) and not np.isnan(ma20) and current_price < ma20 else "Neutral"
        medium_term_trend = "Bullish" if not np.isnan(current_price) and not np.isnan(ma50) and current_price > ma50 else "Bearish" if not np.isnan(current_price) and not np.isnan(ma50) and current_price < ma50 else "Neutral"
        long_term_trend = "Bullish" if not np.isnan(current_price) and not np.isnan(ma200) and current_price > ma200 else "Bearish" if not np.isnan(current_price) and not np.isnan(ma200) and current_price < ma200 else "Neutral"

        print("Calculating volatility...")
         # Volatility calculation moved here if not part of price_levels
        daily_returns = df['Close'].pct_change()
        # Filter out NaNs before std calculation
        valid_returns = daily_returns.dropna()
        annual_volatility = valid_returns.std() * np.sqrt(252) * 100 if not valid_returns.empty else np.nan
         # --- ENSURE VOLATILITY IS SCALAR FLOAT ---
        # Explicitly convert to float, handling potential NaN result
        if isinstance(annual_volatility, (int, float)):
             # If it's already a scalar int/float, use it (checking for NaN)
             volatility = float(annual_volatility) if pd.notna(annual_volatility) else np.nan
        elif isinstance(annual_volatility, pd.Series) and not annual_volatility.empty:
             # If it's a non-empty Series, try to get the item and convert, handle NaN
             # Use .iloc[0] or .item() to get the value from the single-element Series
             try:
                 volatility = float(annual_volatility.iloc[0]) if pd.notna(annual_volatility.iloc[0]) else np.nan
             except Exception: # Catch potential errors if .iloc[0] fails unexpectedly
                 volatility = np.nan
        else:
            # Default to NaN if the calculation resulted in an empty Series or unexpected type
            volatility = np.nan
        # --- END ENSURE SCALAR ---
        volatility = annual_volatility

        print("Calculating support and resistance levels...")
        # Ensure enough data for support/resistance (e.g., more than the window size used inside the functions)
        if len(df) > 50: # Assuming a window of up to 50 days in your placeholder functions
            support_levels = calculate_support_levels(df) # Your function
            resistance_levels = calculate_resistance_levels(df) # Your function
        else:
            print("Warning: Not enough data for robust support/resistance calculation.")
            support_levels = []
            resistance_levels = []


        # Get the highest support level below current price, or lowest if all are above
        current_support = None
        if support_levels and not np.isnan(current_price):
            supports_below_price = [s for s in support_levels if s < current_price]
            if supports_below_price:
                 current_support = max(supports_below_price)
            elif support_levels: # If all supports are above current price, consider the lowest one
                 current_support = min(support_levels)

        # Ensure we have necessary values and they are not NaN
        if current_price is not None and current_atr is not None and price_levels.get('target_price') is not None and price_levels.get('stop_loss') is not None:
            # Calculate Entry Range based on Current Price, ATR, and Stop Loss
            # Entry range: From max(Stop Loss, Current Price - 0.5*ATR) to Current Price + 0.1*ATR
            # This suggests an entry area that avoids going below the stop loss
            entry_range_lower = max(price_levels['stop_loss'], current_price - 0.5 * current_atr)
            entry_range_upper = current_price + 0.1 * current_atr

            # Calculate Exit Range based on Target Price and ATR
            # Exit range: From Target Price - ATR to Target Price
            # Ensure target is above current price for a valid exit range for a long position
            target_price = price_levels['target_price']
            if target_price > current_price:
                exit_range_lower = target_price - current_atr
                exit_range_upper = target_price

            # Ensure lower bound of ranges are not higher than upper bound after calculations
            if entry_range_lower is not None and entry_range_upper is not None:
                 entry_range_lower = min(entry_range_lower, entry_range_upper) # Correct if lower somehow ends up higher
                 entry_range_upper = max(entry_range_lower, entry_range_upper) # Ensure upper is max

            if exit_range_lower is not None and exit_range_upper is not None:
                 exit_range_lower = min(exit_range_lower, exit_range_upper) # Correct if lower somehow ends up higher
                 exit_range_upper = max(exit_range_lower, exit_range_upper) # Ensure upper is max
        # --- Check Candlestick Patterns ---
        print("Checking candlestick patterns...")
        patterns = check_candlestick_patterns(df)       

        # --- Get News Sentiment ---
        print("Getting news sentiment...")
        news_data = get_news_sentiment(ticker)
        sentiment_score = news_data.get('sentiment_score', 0.5) # Default score to 0.5 (Neutral)
        sentiment_label = news_data.get('sentiment_label', 'Neutral')


        # --- Define Boolean Conditions (Still useful for summary/override) ---
        print("Defining conditions...")
        # --- FINAL SAFEGUARD: Ensure volatility is a scalar right before use ---
        # This block explicitly converts 'volatility' to a scalar float
        # just before it's used in the conditions dictionary, handling potential NaNs.
        final_volatility = np.nan # Initialize as NaN

        # Check the type of the 'volatility' variable calculated earlier
        if isinstance(volatility, (int, float)):
             # If it's already a scalar int/float, use it (checking for NaN)
             final_volatility = float(volatility) if pd.notna(volatility) else np.nan
        elif isinstance(volatility, pd.Series) and not volatility.empty:
             # If it's a non-empty Series, try to get the item and convert, handle NaN
             try:
                 # Use .iloc[0] or .item() to get the value from the single-element Series
                 value = volatility.iloc[0] # Get the single value
                 final_volatility = float(value) if pd.notna(value) else np.nan
             except Exception: # Catch potential errors if .iloc[0] fails unexpectedly
                 final_volatility = np.nan
        # else: volatility remains np.nan as initialized

        conditions = {
            # Trend Conditions
            "Strong Uptrend": not np.isnan(ma200) and current_price > ma20 and current_price > ma50 and current_price > ma200,
            "Price > MA20": not np.isnan(ma20) and current_price > ma20,
            "Price > MA50": not np.isnan(ma50) and current_price > ma50,
            "Price > MA200": not np.isnan(ma200) and current_price > ma200,

            # Momentum Conditions
            "RSI Bullish": not np.isnan(rsi) and 40 <= rsi <= 70,
            "RSI Oversold": not np.isnan(rsi) and rsi < 30,
            "RSI Overbought": not np.isnan(rsi) and rsi > 70,

            "MACD Bullish": not np.isnan(macd) and not np.isnan(macd_signal) and macd > macd_signal and macd > 0,
            "MACD Bearish": not np.isnan(macd) and not np.isnan(macd_signal) and (macd < macd_signal or macd < 0),

            "Stochastic Bullish": not np.isnan(stoch_k) and not np.isnan(stoch_d) and stoch_k > stoch_d and stoch_k <= 80,
             "Stochastic Oversold": not np.isnan(stoch_k) and stoch_k < 20,
            "Stochastic Overbought": not np.isnan(stoch_k) and stoch_k > 80,

            # Volume Conditions
            "Strong Volume": not np.isnan(volume) and not np.isnan(volume_ma20) and volume_ma20 > 0 and volume / volume_ma20 > 1.5, # Check MA > 0
            "Increasing Volume": not np.isnan(volume_change) and volume_change > 20, # Percentage change > 20%
            
            # Mondey Flow
            "Positive CMF": not np.isnan(cmf) and cmf > 0,
            "Negative CMF": not np.isnan(cmf) and cmf < 0,

            # Pattern Conditions (Based on findings from check_candlestick_patterns)
            "Bullish Pattern": any(p in [
                "Hammer (Bullish Reversal Shape)", # Updated name
                "Morning Star",
                "Bullish Engulfing",
                "Bullish Harami",
                "Dragonfly Doji",
                "Bullish Marubozu",       # Added
                "Inverted Hammer (Bullish Reversal Shape)", # Added
                "Piercing Pattern"        # Added
                 # Add other bullish patterns here
                ] for p in patterns),

            "Bearish Pattern": any(p in [
                "Hanging Man (Bearish Reversal Shape)", # Updated name
                "Evening Star",
                "Bearish Engulfing",
                "Bearish Harami",
                "Shooting Star (Bearish Reversal Shape)", # Updated name
                "Shooting Star Doji",
                "Bearish Marubozu",       # Added
                "Dark Cloud Cover"        # Added
                # Add other bearish patterns here
                ] for p in patterns),

            # Neutral Pattern: True if any pattern is found, but none are classified as bullish or bearish above
            "Neutral Pattern": bool(patterns) and not any(p in patterns for p in [
                "Hammer (Bullish Reversal Shape)", "Morning Star", "Bullish Engulfing", "Bullish Harami", "Dragonfly Doji",
                "Bullish Marubozu", "Inverted Hammer (Bullish Reversal Shape)", "Piercing Pattern", # Bullish
                "Hanging Man (Bearish Reversal Shape)", "Evening Star", "Bearish Engulfing", "Bearish Harami", "Shooting Star (Bearish Reversal Shape)",
                "Shooting Star Doji", "Bearish Marubozu", "Dark Cloud Cover" # Bearish
                # Add other classified patterns here
                ]),

            # Risk Management
            "Good Risk-Reward": risk_reward_ratio >= 2.0,
            "Reasonable Volatility": not np.isnan(final_volatility) and final_volatility >= 15 and final_volatility <= 40,

            # Market Sentiment (Using the label for the boolean condition)
            "Positive News": sentiment_label == "Positive",

            # Support/Resistance
             "Above Support": not np.isnan(current_price) and current_support is not None and current_price > current_support
        }

        # --- Calculate Dynamic Strength Score (Point System Revised) ---
        print("Calculating dynamic strength score...")
        raw_score = 0

        # --- Assign Dynamic Points based on Indicator Values ---

        # Trend Points (More points the further above MAs)
        if not np.isnan(current_price):
            if not np.isnan(ma20) and ma20 > 0:
                diff_ma20 = (current_price - ma20) / ma20 * 100 # Percentage difference
                raw_score += min(max(round(diff_ma20 * 0.2), -3), 3) # e.g., +/- 0.2 points per %, max +/- 3
            if not np.isnan(ma50) and ma50 > 0:
                 diff_ma50 = (current_price - ma50) / ma50 * 100
                 raw_score += min(max(round(diff_ma50 * 0.15), -3), 3) # Slightly less weight than MA20
            if not np.isnan(ma200) and ma200 > 0:
                 diff_ma200 = (current_price - ma200) / ma200 * 100
                 raw_score += min(max(round(diff_ma200 * 0.1), -4), 4) # Longer term, slightly more max impact


        # Momentum Points (Points based on position within ranges)
        if not np.isnan(rsi):
            if rsi > 50: # Bullish zone
                raw_score += min(round((rsi - 50) * 0.3), 5) # e.g., 0.3 points per point above 50, max 5
            elif rsi < 50: # Bearish zone
                 raw_score += max(round((rsi - 50) * 0.3), -5) # e.g., 0.3 points per point below 50, min -5

        if not np.isnan(macd) and not np.isnan(macd_signal):
            macd_diff = macd - macd_signal
            if macd > 0 and macd_diff > 0: # Bullish crossover above zero
                raw_score += min(round(macd_diff * 5), 4) # e.g., 5 points per unit diff, max 4 (adjust scaling based on typical MACD values)
            elif macd < 0 and macd_diff < 0: # Bearish crossover below zero
                 raw_score += max(round(macd_diff * 5), -4) # e.g., 5 points per unit diff, min -4
            elif macd_diff > 0: # Bullish crossover below zero
                 raw_score += min(round(macd_diff * 3), 2) # Less strong signal
            elif macd_diff < 0: # Bearish crossover above zero
                 raw_score += max(round(macd_diff * 3), -2) # Less strong signal


        if not np.isnan(stoch_k) and not np.isnan(stoch_d):
             stoch_k_diff = stoch_k - stoch_d
             if stoch_k <= 80 and stoch_d <= 80: # Not in overbought
                 if stoch_k_diff > 0: # Bullish cross/position
                      raw_score += min(round(stoch_k_diff * 0.5), 3) # e.g., 0.5 points per diff, max 3
                 elif stoch_k_diff < 0: # Bearish cross/position
                      raw_score += max(round(stoch_k_diff * 0.5), -3) # e.g., 0.5 points per diff, min -3

             # Add specific points for entering/exiting overbought/oversold
             if stoch_k > 80 and stoch_d <= 80: raw_score -= 2 # Entering overbought
             elif stoch_k < 20 and stoch_d >= 20: raw_score += 2 # Entering oversold (potential bounce)


        # Volume Points (More points for higher volume relative to MA)
        if not np.isnan(volume) and not np.isnan(volume_ma20) and volume_ma20 > 0:
            volume_ratio = volume / volume_ma20
            if volume_ratio > 1.0:
                 raw_score += min(round((volume_ratio - 1.0) * 2), 3) # e.g., 2 points per 1x MA above 1, max 3

        # CMF Points
        if not np.isnan(cmf):
            if cmf > 0:  # Positive CMF (accumulation)
                raw_score += 2
            elif cmf < 0:  # Negative CMF (distribution)
                raw_score -= 2

        # Pattern Points (Can still keep these fixed or add more nuance)
        if conditions.get("Bullish Pattern"): raw_score += 4
        if conditions.get("Bearish Pattern"): raw_score -= 5
        # Neutral patterns add 0

        # Risk Management Points (Points based on Risk:Reward ratio)
        if risk_reward_ratio > 1.0: # Only consider if R:R is positive
            raw_score += min(round(risk_reward_ratio * 1.5), 5) # e.g., 1.5 points per R:R ratio point, max 5

        # Volatility Points (Penalize very high, reward reasonable)
        if not np.isnan(final_volatility):
            if final_volatility > 40: raw_score -= 2 # Penalize high volatility
            elif final_volatility >= 15 and final_volatility <= 40: raw_score += 1 # Reward reasonable volatility


        # Market Sentiment Points (Points based on the numerical sentiment score)
        if not np.isnan(sentiment_score):
             # Scale score from [0, 1] or [-1, 1] to points. Assuming [0, 1] where 0.5 is neutral.
             # Points range from -3 to +3
             raw_score += min(max(round((sentiment_score - 0.5) * 10), -3), 3) # e.g., 10 points per 1.0 score difference from 0.5, max +/-3


        # Support/Resistance Points (Simple check)
        if conditions.get("Above Support"): raw_score += 1
        # Could add points for bouncing off support, breaking resistance etc.


        # --- Determine Min/Max Possible Raw Scores for Normalization ---
        # This requires recalculating based on the *new* point ranges
        # Max possible positive raw score: Sum of max points from each dynamic/static positive contributor
        max_raw_pos = sum([
            3 + 3 + 4, # Max from MA points (3 MAs)
            5,         # Max from RSI
            4,         # Max from MACD
            3 + 2,     # Max from Stoch (cross + oversold entry)
            3,         # Max from Volume Ratio
            4,         # Bullish Pattern
            5,         # Max from Risk:Reward
            1,         # Reasonable Volatility
            3,         # Max from Sentiment
            1          # Above Support
            # Total max positive = 3+3+4 + 5 + 4 + 3+2 + 3 + 4 + 5 + 1 + 3 + 1 = 38 + 1 = 39. Let's re-calculate carefully.
            # MA points: max 3+3+4=10
            # Momentum: max 5 (RSI) + 4 (MACD) + 3+2 (Stoch) = 14
            # Volume: max 3
            # Pattern: max 4 (Bullish)
            # Risk: max 5 (R:R) + 1 (Volatility) = 6
            # Sentiment: max 3
            # Support: max 1
            # Total Max Positive = 10 + 14 + 3 + 4 + 6 + 3 + 1 = 41
            # Let's verify against the code logic - the code uses min/max() which caps points
            # MA points: max 3+3+4=10
            # RSI: max 5
            # MACD: max 4
            # Stoch: max 3+2 = 5 (cross + oversold entry)
            # Volume: max 3
            # Bullish Pattern: 4
            # Risk: max 5
            # Reasonable Volatility: 1
            # Sentiment: max 3
            # Above Support: 1
            # Total Max Positive = 10 + 5 + 4 + 5 + 3 + 4 + 5 + 1 + 3 + 1 = 41
        ])
        max_possible_score = 41


        # Min possible negative raw score: Sum of min points from each dynamic/static negative contributor
        min_raw_neg = sum([
            -3 + -3 + -4, # Min from MA points
            -5,           # Min from RSI
            -4,           # Min from MACD
            -3,           # Min from Stoch
            0,            # Volume doesn't subtract points in this logic
            -5,           # Bearish Pattern
            0,            # Risk:Reward doesn't subtract points
            -2,           # High Volatility
            -3,           # Min from Sentiment
            0             # Support doesn't subtract points
            # Total min negative = -3-3-4 -5 -4 -3 -5 -2 -3 = -10 -5 -4 -3 -5 -2 -3 = -32
            # Let's verify against the code logic caps
            # MA points: min -3 -3 -4 = -10
            # RSI: min -5
            # MACD: min -4
            # Stoch: min -3
            # Bearish Pattern: -5
            # High Volatility: -2
            # Sentiment: min -3
            # Total Min Negative = -10 -5 -4 -3 -5 -2 -3 = -32
         ])
        min_possible_score = -32


        # Normalize the raw score to a 0-100 scale
        range_score = max_possible_score - min_possible_score
        if range_score > 0:
             strength_score = ((raw_score - min_possible_score) / range_score) * 100
        else:
             strength_score = 50 # Default to neutral if range is zero

        # Clamp score between 0 and 100 just in case
        strength_score = max(0, min(100, strength_score))
        strength_score = round(strength_score, 2)

        # --- End Dynamic Strength Score Calculation ---


        print("Creating result dictionary...")
        # Ensure numerical values are not NaN before rounding
        result = {
            "ticker": ticker,
            "price": round(current_price, 2) if not np.isnan(current_price) else None,
            "cmf": round(cmf, 2) if not np.isnan(cmf) else None,  # Include CMF
            # Add the new range values (formatted as strings for display)
            "entry_range": f"₹{entry_range_lower:.2f} - ₹{entry_range_upper:.2f}" if entry_range_lower is not None and entry_range_upper is not None else "N/A",
            "exit_range": f"₹{exit_range_lower:.2f} - ₹{exit_range_upper:.2f}" if exit_range_lower is not None and exit_range_upper is not None else "N/A",
            "target": price_levels.get('target_price', None),
            "stop_loss": price_levels.get('stop_loss', None),
            "daily_change": round(daily_change, 2),
            "target": price_levels.get('target_price', None),
            "stop_loss": price_levels.get('stop_loss', None),
            "risk_reward": price_levels.get('risk_reward_ratio', None),
            "support_levels": [round(s, 2) for s in support_levels[:3]] if support_levels else [], # Ensure rounding
            "resistance_levels": [round(r, 2) for r in resistance_levels[:3]] if resistance_levels else [], # Ensure rounding
            "rsi": round(rsi, 2) if not np.isnan(rsi) else None,
            "macd": round(macd, 4) if not np.isnan(macd) else None,
            "stoch_k": round(stoch_k, 2) if not np.isnan(stoch_k) else None,
            "stoch_d": round(stoch_d, 2) if not np.isnan(stoch_d) else None,
            "atr": price_levels.get('atr', None),
            "volatility": round(final_volatility, 2) if not np.isnan(final_volatility) else None,
            "volume": volume if not np.isnan(volume) else None,
            "volume_change": round(volume_change, 2),
            "volume_ma20": volume_ma20 if not np.isnan(volume_ma20) else None,
            "short_term_trend": short_term_trend,
            "medium_term_trend": medium_term_trend,
            "long_term_trend": long_term_trend,
            "candlestick_patterns": patterns,
            "news_sentiment": sentiment_label,
            "sentiment_score": round(sentiment_score, 2) if not np.isnan(sentiment_score) else None,
            "latest_news": news_data.get("latest_news", []),
            "conditions": conditions, # Still keep boolean conditions
            # conditions_met and total_conditions are less relevant with point system
            "raw_strength_score": raw_score, # Add raw score for debugging/understanding
            "strength_score": strength_score, # Use the new normalized score
            "analysis_summary": generate_analysis_summary(conditions, patterns, news_data, raw_score, strength_score), 
            "trade_recommendation": generate_trade_recommendation(strength_score, conditions, raw_score),
            "data": df
        }

        print(f"Analysis completed for {ticker}")
        return result

    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        print("Full error traceback:")  
        print(traceback.format_exc())  # Print the full traceback for debugging     
        return None

def scan_bullish_stocks(stocks_dict):
    """Scan all Nifty 200 stocks and return those with strength score > 66%"""
    bullish_stocks = []
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Track progress
    total_stocks = len(stocks_dict)
    progress = 0
    
    for stock_name, ticker in stocks_dict.items():
        try:
            # Show progress
            progress += 1
            progress_bar.progress(progress / total_stocks)
            status_text.text(f"Scanning {stock_name} ({progress}/{total_stocks})")
            
            # Analyze the stock
            result = analyze_stock(ticker)
            
            if result and result['strength_score'] > 66:
                bullish_stocks.append({
                    'stock_name': stock_name,
                    'ticker': ticker,
                    'strength_score': result['strength_score'],
                    'current_price': result['price'],
                    'daily_change': result['daily_change'],
                    'trade_recommendation': result['trade_recommendation'],
                    'target': result['target'],
                    'stop_loss': result['stop_loss'],
                    'risk_reward': result['risk_reward'],
                    'support_levels': result['support_levels'],
                    'resistance_levels': result['resistance_levels']
                })
                
        except Exception as e:
            print(f"Error analyzing {stock_name}: {str(e)}")
    
    # Sort by strength score descending
    bullish_stocks.sort(key=lambda x: x['strength_score'], reverse=True)
    
    # Reset progress bar
    progress_bar.empty()
    status_text.empty()
    
    return bullish_stocks

def display_bullish_stocks(bullish_stocks):
    """Display the bullish stocks in a nice table format with target and SL"""
    if not bullish_stocks:
        st.info("No bullish stocks found with strength score > 66%")
        return
    
    st.markdown("<h2 class='section-title'>Bullish Stocks (Strength Score > 66%)</h2>", unsafe_allow_html=True)
    
    # Create a table to display results
    data = []
    for stock in bullish_stocks:
        data.append({
            'Stock': stock['stock_name'],
            'Ticker': stock['ticker'],
            'Current Price': f"₹{stock['current_price']:,}",
            'Daily Change': f"{stock['daily_change']}%",
            'Strength Score': f"{stock['strength_score']}%",
            'Recommendation': stock['trade_recommendation'],
            'Target Price': f"₹{stock['target']:,}",
            'Stop Loss': f"₹{stock['stop_loss']:,}",
            'Risk-Reward Ratio': f"{stock['risk_reward']:.2f}",
            'Support Levels': ', '.join(map(lambda x: f"₹{x:,}", stock['support_levels'])),
            'Resistance Levels': ', '.join(map(lambda x: f"₹{x:,}", stock['resistance_levels']))
        })
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Add color coding for positive/negative changes
    def color_code_change(val):
        color = 'green' if float(val.split('%')[0]) > 0 else 'red'
        return f'color: {color}'
    
    # Add color coding for risk-reward ratio
    def color_code_risk_reward(val):
        ratio = float(val)
        if ratio >= 2.0:
            return 'color: green'
        elif ratio >= 1.5:
            return 'color: orange'
        else:
            return 'color: red'
    
    # Add color coding for strength score
    def color_code_strength(val):
        score = float(val.split('%')[0])
        if score >= 90:
            return 'color: green'
        elif score >= 70:
            return 'color: orange'
        else:
            return 'color: blue'
    
    # Style the DataFrame
    styled_df = df.style.applymap(color_code_change, subset=['Daily Change']) \
                       .applymap(color_code_risk_reward, subset=['Risk-Reward Ratio']) \
                       .applymap(color_code_strength, subset=['Strength Score'])
    
    # Display the table with sorting
    st.dataframe(styled_df, use_container_width=True)
    
    # Add summary statistics
    st.markdown("""
    <div class="metric-card">
        <h3>Summary</h3>
        <p>Total Bullish Stocks: {total_stocks}</p>
        <p>Average Strength Score: {avg_strength:.2f}%</p>
        <p>Average Risk-Reward Ratio: {avg_risk_reward:.2f}</p>
        <p>Number of Stocks with R/R > 2.0: {high_rr_count}</p>
    </div>
    """.format(
        total_stocks=len(bullish_stocks),
        avg_strength=sum(stock['strength_score'] for stock in bullish_stocks) / len(bullish_stocks),
        avg_risk_reward=sum(stock['risk_reward'] for stock in bullish_stocks) / len(bullish_stocks),
        high_rr_count=sum(1 for stock in bullish_stocks if stock['risk_reward'] >= 2.0)
    ), unsafe_allow_html=True)
    
    # Add download button for CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Bullish Stocks Data",
        data=csv,
        file_name='bullish_stocks.csv',
        mime='text/csv'
    )
 
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Stock Analysis Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"        
    )
  
     # Get the current path from URL
    path = st.query_params.get("path", ["login"])[0]
    
    # Check if user is authenticated
    authenticated = st.session_state.get("authenticated", False)
    
    # Force login if not authenticated
    if not authenticated:
        # Hide sidebar for login page
        st.markdown("""
            <style>
                section[data-testid="stSidebar"] {
                    display: none;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Add login page styling
        st.markdown("""
        <style>
            .login-container {
                max-width: 400px;
                margin: 0 auto;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                background-color: white;
            }
        </style>
        """, unsafe_allow_html=True)       
        
        st.title("Login")
        st.write("Please enter your credentials to access the dashboard.")
        check_password()
        st.markdown('</div>', unsafe_allow_html=True)       
    
            
    else:  # dashboard page
        # Your sidebar code here       
        
        st.title("Stock Analysis Dashboard")
        st.write("Welcome to the dashboard!")
       
        # Add buttons for scanning and backtesting
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Scan Bullish Stocks", use_container_width=True):
        # Get Nifty 200 stocks
        nifty_200_stocks = get_nifty_200_stocks()
        
        # Add loading spinner
        with st.spinner("Scanning Nifty 200 stocks for bullish patterns..."):
            bullish_stocks = scan_bullish_stocks(nifty_200_stocks)
            
            if bullish_stocks:
                display_bullish_stocks(bullish_stocks)
            else:
                st.info("No bullish stocks found with strength score > 66%")    
    
    # Individual stock analysis section
    st.sidebar.markdown("---")
    st.sidebar.header("Individual Stock Analysis")
    
    # Get Nifty 200 stocks (hardcoded)
    nifty_200_stocks = get_nifty_200_stocks()
    
    # Create dropdown with stocks
    selected_stock = st.sidebar.selectbox(
        "Select Nifty 200 Stock",
        options=["Select a stock..."] + list(nifty_200_stocks.keys())
    )
    
    # Only proceed if a stock is selected
    if selected_stock != "Select a stock...":
        # Get the selected ticker
        ticker = nifty_200_stocks[selected_stock]
        
        # Analysis button with better styling
        if st.sidebar.button("Analyze Stock", use_container_width=True):
            if not ticker:
                st.error("Please enter a stock ticker")
                return
                
            with st.spinner("Analyzing stock..."):
                result = analyze_stock(ticker)
                
                if result is None:
                    st.error(f"Failed to analyze {ticker}. Please check the ticker and try again.")
                    return
                
            # Create tabs with improved styling
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Technical Analysis", "📊 Charts", "📰 News"])
            
            with tab1:
                st.markdown("<h2 class='section-title'>Stock Overview</h2>", unsafe_allow_html=True)
                
                # Create metric cards grid
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Current Price</h3>
                        <p>₹{result['price']:,}</p>
                        <p class="{'positive' if result['daily_change'] >= 0 else 'negative'}">
                            {result['daily_change']}%
                            <span class="{'bullish' if result['daily_change'] >= 0 else 'bearish'}">
                                {'↑' if result['daily_change'] >= 0 else '↓'}
                            </span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Technical Indicators</h3>
                        <p>RSI: {result['rsi']:.2f}</p>
                        <p>MACD: {result['macd']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add CMF here
                        
                    if not np.isnan(result['cmf']):
                        cmf_color = 'green' if result['cmf'] > 0 else 'red'
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Chaikin Money Flow (CMF)</h3>
                            <p style="color: {cmf_color};">{result['cmf']:.2f}</p>
                            <p style="font-size: 0.8em; color: #555;">Indicates buying/selling pressure.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="metric-card">
                            <h3>Chaikin Money Flow (CMF)</h3>
                            <p>N/A</p>
                            <p style="font-size: 0.8em; color: #555;">Indicates buying/selling pressure.</p>
                        </div>
                        """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Volume</h3>
                        <p>{result['volume']:,}</p>
                        <p class="{'positive' if result['volume_change'] >= 0 else 'negative'}">
                            {result['volume_change']}% change
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Trend Analysis with animated arrows
                st.markdown("<h2 class='section-title'>Trend Analysis</h2>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Short-term</h3>
                        <p class="{'bullish' if result['short_term_trend'] == 'Bullish' else 'bearish'}">
                            {result['short_term_trend']}
                            <span class="trend-arrow">{'↑' if result['short_term_trend'] == 'Bullish' else '↓'}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Medium-term</h3>
                        <p class="{'bullish' if result['medium_term_trend'] == 'Bullish' else 'bearish'}">
                            {result['medium_term_trend']}
                            <span class="trend-arrow">{'↑' if result['medium_term_trend'] == 'Bullish' else '↓'}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Long-term</h3>
                        <p class="{'bullish' if result['long_term_trend'] == 'Bullish' else 'bearish'}">
                            {result['long_term_trend']}
                            <span class="trend-arrow">{'↑' if result['long_term_trend'] == 'Bullish' else '↓'}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                 # --- Add new row for Entry/Exit Ranges and Stop Loss ---
               
                st.markdown("---") # Add a separator
                col4, col5, col6 = st.columns(3)

                with col4:
                    # Entry Range Card
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Best Entry Range</h3>
                        <p>{result.get('entry_range', 'N/A')}</p>
                        <p style="font-size: 0.8em; color: #555;">Suggested entry area.</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col5:
                    # Exit Range Card
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Potential Exit Range</h3>
                        <p>{result.get('exit_range', 'N/A')}</p>
                        <p style="font-size: 0.8em; color: #555;">Suggested area to consider exiting.</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col6:
                    # Stop Loss Card (Displaying the single calculated value)
                    # Add color coding if you want (e.g., neutral as it's a risk management point)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Stop Loss</h3>
                        <p>₹{result.get('stop_loss', 'N/A'):,.2f}</p>
                         <p style="font-size: 0.8em; color: #555;">Level to limit potential losses.</p>
                    </div>
                    """, unsafe_allow_html=True)
                # Support/Resistance with gradient background
                st.markdown("<h2 class='section-title'>Support/Resistance Levels</h2>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="section-title">Support Levels</div>
                    <p>{', '.join(map(lambda x: f"₹{x:,}", result['support_levels']))}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="section-title">Resistance Levels</div>
                    <p>{', '.join(map(lambda x: f"₹{x:,}", result['resistance_levels']))}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Trade Recommendation with animation
                st.markdown("<h2 class='section-title'>Trade Recommendation</h2>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="recommendation-box">
                    <h2>{result['trade_recommendation']}</h2>
                    <div class="strength-meter">
                        <div class="strength-meter-fill" style="width: {result['strength_score']}%"></div>
                    </div>
                    <p>Strength Score: {result['strength_score']}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown("<h2 class='section-title'>Technical Indicators</h2>", unsafe_allow_html=True)
                
                # Conditions grid with hover effects
                col1, col2 = st.columns(2)
                with col1:
                    for condition, met in list(result['conditions'].items())[:len(result['conditions'])//2]:
                        status = "✅" if met else "❌"
                        color = "positive" if met else "negative"
                        st.markdown(f"""
                        <div class="condition-card">
                            <p class="{color}">{status} {condition}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    for condition, met in list(result['conditions'].items())[len(result['conditions'])//2:]:
                        status = "✅" if met else "❌"
                        color = "positive" if met else "negative"
                        st.markdown(f"""
                        <div class="condition-card">
                            <p class="{color}">{status} {condition}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Candlestick patterns
                st.markdown("<h2 class='section-title'>Candlestick Patterns</h2>", unsafe_allow_html=True)
                if result['candlestick_patterns']:
                    for pattern in result['candlestick_patterns']:
                        # Get the description, defaulting to a message if not found
                        description = PATTERN_DESCRIPTIONS.get(pattern, "Description not available.")
                        # Determine color based on pattern name for consistency
                        color_class = 'positive' if any(word in pattern for word in ['Bullish', 'Hammer', 'Dragonfly', 'Morning', 'Piercing']) else \
                                      'negative' if any(word in pattern for word in ['Bearish', 'Hanging Man', 'Shooting Star', 'Evening', 'Dark Cloud']) else 'neutral'

                        st.markdown(f"""
                        <div class="pattern-card">
                            <p class="{color_class}" style="font-weight: bold; margin-bottom: 5px;">
                                {pattern}
                            </p>
                            <p style="font-size: 0.9em; color: #555;">
                                {description}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="pattern-card">
                        <p class="neutral">No significant patterns detected recently.</p>
                    </div>
                    """, unsafe_allow_html=True)

                
               
                st.markdown("""
                <style>
                .summary-card {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }
                .summary-card ul {
                    list-style-type: disc;
                    margin-left: 20px;
                    padding: 10px 0;
                }
                .summary-card li {
                    margin: 8px 0;
                    font-size: 16px;
                    line-height: 1.5;
                }
                </style>
                """, unsafe_allow_html=True)

                # Split the summary into lines and display as a list
                st.markdown("<h2 class='section-title'>Analysis Summary</h2>", unsafe_allow_html=True)
                # The analysis_summary string now contains markdown list formatting and HTML spans
                st.markdown(result['analysis_summary'], unsafe_allow_html=True)                
            
            with tab3:
                st.markdown("<h2 class='section-title'>Price Charts</h2>", unsafe_allow_html=True)
                
                # Candlestick chart with improved styling
                fig = go.Figure(data=[go.Candlestick(x=result['data'].index,
                                                     open=result['data']['Open'],
                                                     high=result['data']['High'],
                                                     low=result['data']['Low'],
                                                     close=result['data']['Close'])])
                
                # Add moving averages with different colors
                fig.add_trace(go.Scatter(x=result['data'].index, 
                                        y=result['data']['MA20'],
                                        name='MA20',
                                        line=dict(color='#1e90ff', width=1)))
                
                fig.add_trace(go.Scatter(x=result['data'].index, 
                                        y=result['data']['MA50'],
                                        name='MA50',
                                        line=dict(color='#22c55e', width=1)))
                
                fig.update_layout(
                    title=f"{result['ticker']} Price Chart",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    height=600,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Arial", size=12, color="#6b7280"),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart with improved styling
                st.markdown("<h2 class='section-title'>Volume Analysis</h2>", unsafe_allow_html=True)
                volume_fig = go.Figure(data=[go.Bar(x=result['data'].index,
                                                    y=result['data']['Volume'],
                                                    name='Volume')])
                
                volume_fig.update_layout(
                    title="Volume Chart",
                    xaxis_title='Date',
                    yaxis_title='Volume',
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Arial", size=12, color="#6b7280"),
                    hovermode='x unified'
                )
                
                st.plotly_chart(volume_fig, use_container_width=True)
            
            with tab4:
                st.markdown("<h2 class='section-title'>News Sentiment</h2>", unsafe_allow_html=True)
                if result['latest_news']:
                    for news in result['latest_news']:
                        st.markdown(f"""
                        <div class="news-card">
                            <h3>{news.get('title', 'No title')}</h3>
                            <p>{news.get('description', 'No description available')}</p>
                            <div class="news-meta">
                              <a href="{news.get('sourceLink', '#')}"><span class="neutral">Source: {news.get('sourceTitle', 'Unknown')}</span></a>
                                <div class="neutral">Date: {news.get('date', 'Unknown')}</div>
                                <div class="{'positive' if news.get('sentiment', '') == 'Positive' else 'negative'}">
                                    Sentiment: {news.get('sentiment', 'Unknown')}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Overall sentiment summary
                st.markdown("<h2 class='section-title'>Overall Sentiment</h2>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="section-title">Sentiment Analysis</div>
                    <p>Score: {result['sentiment_score']:.2f}</p>
                    <p class="{'positive' if result['news_sentiment'] == 'Positive' else 'negative'}">
                        {result['news_sentiment']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
     # Add footer with copyright
        current_year = datetime.now().year
        st.markdown(f"""
        <div class="footer">
            © {current_year} Kalyan Thumu. All rights reserved.
        </div>
        """, unsafe_allow_html=True)
        # Add disclaimer text
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <strong>Disclaimer:</strong>
            <ul style='margin-top: 10px;'>
                <li>This tool uses technical indicators and price action patterns to generate predictions.</li>
                <li>Please understand that these methods are based on probability, not certainty.</li>
                <li>There is no guarantee that the predictions will be accurate.</li>
                <li>Financial risk is involved in all trading activities.</li>
                <li>I am not a SEBI-registered advisor, and this is not financial advice.</li>
                <li>Always do your own research or consult a certified financial advisor before making any investment decisions.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
                # Add custom CSS for styling
    st.markdown("""
    <style>
    .stAppToolbar {
                display: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 1rem 2rem;
        border-radius: 1rem;
        background-color: #f0f2f6;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e2e6;
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"]:active {
        background-color: #1e90ff;
        color: white;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6 0%, #e0e2e6 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .positive {
        color: #22c55e;
    }
    
    .negative {
        color: #ef4444;
    }
    
    .neutral {
        color: #6b7280;
    }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .condition-card {
        background: white;
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .condition-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .pattern-card {
        background: white;
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .news-card {
        background: white;
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .strength-meter {
        height: 10px;
        border-radius: 5px;
        background: #e5e7eb;
        overflow: hidden;
    }
    
    .strength-meter-fill {
        height: 100%;
        background: linear-gradient(90deg, #1e90ff 0%, #22c55e 100%);
        transition: width 0.5s ease;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #1e90ff 0%, #22c55e 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .trend-arrow {
        font-size: 1.2rem;
        margin-left: 0.5rem;
    }
    
    .bullish {
        color: #22c55e;
    }
    
    .bearish {
        color: #ef4444;
    }
    
    .summary-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1e90ff;
    }
    
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
