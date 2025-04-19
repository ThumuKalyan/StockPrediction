import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
from gnews import GNews

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
        
        return df
    except Exception as e:
        st.error(f"Error in calculate_technical_indicators: {str(e)}")
        return None

def calculate_rsi(data, periods=14):
    """Calculate RSI using manual calculation"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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
        
        return df
    except Exception as e:
        st.error(f"Error in calculate_technical_indicators: {str(e)}")
        return None

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
        # Remove .NS suffix and handle special cases
        company_name = ticker.replace('.NS', '')
        
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
            sentiment = blob.sentiment.polarity
            sentiments.append(sentiment)
            
            # Store news details
            latest_news.append({
                "title": item['title'],
                "published": item['published date'],
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

def generate_analysis_summary(conditions, patterns, news_data):
    """Generate a comprehensive analysis summary based on technical and fundamental indicators"""
    summary = []
    
    # Trend Analysis
    trend_summary = []
    if conditions["Strong Uptrend"]:
        trend_summary.append("Strong uptrend across all timeframes")
    elif conditions["Price > MA20"] and conditions["Price > MA50"]:
        trend_summary.append("Bullish trend on medium-term")
    else:
        trend_summary.append("Mixed trend conditions")
    summary.append(f"Trend: {', '.join(trend_summary)}")
    
    # Momentum Analysis
    momentum_summary = []
    if conditions["RSI Bullish"]:
        momentum_summary.append("RSI indicates bullish momentum")
    if conditions["MACD Bullish"]:
        momentum_summary.append("MACD shows bullish crossover")
    if conditions["Stochastic Bullish"]:
        momentum_summary.append("Stochastic indicates bullish momentum")
    summary.append(f"Momentum: {', '.join(momentum_summary)}")
    
    # Pattern Analysis
    pattern_summary = []
    if conditions["Bullish Pattern"]:
        pattern_summary.append(f"Bullish candlestick patterns: {', '.join(patterns)}")
    elif conditions["Bearish Pattern"]:
        pattern_summary.append(f"Bearish candlestick patterns: {', '.join(patterns)}")
    else:
        pattern_summary.append("No significant patterns detected")
    summary.append(f"Patterns: {', '.join(pattern_summary)}")
    
    # Sentiment Analysis
    sentiment_summary = []
    if conditions["Positive News"]:
        sentiment_summary.append(f"Positive news sentiment (Score: {news_data['sentiment_score']:.2f})")
    else:
        sentiment_summary.append(f"Neutral/Negative news sentiment (Score: {news_data['sentiment_score']:.2f})")
    summary.append(f"Sentiment: {', '.join(sentiment_summary)}")
    
    # Risk Management
    risk_summary = []
    if conditions["Good Risk-Reward"]:
        risk_summary.append("Favorable risk-reward ratio")
    if conditions["Reasonable Volatility"]:
        risk_summary.append("Moderate volatility levels")
    summary.append(f"Risk: {', '.join(risk_summary)}")
    
    return "\n".join(summary)

def generate_trade_recommendation(strength_score, conditions):
    """Generate a trade recommendation based on analysis results"""
    if strength_score >= 80:
        return "STRONG BUY - Multiple bullish conditions aligned"
    
    if strength_score >= 60:
        reasons = []
        if conditions["Strong Uptrend"]:
            reasons.append("Strong trend")
        if conditions["RSI Bullish"] and conditions["MACD Bullish"]:
            reasons.append("Strong momentum")
        if conditions["Bullish Pattern"]:
            reasons.append("Bullish patterns")
        if conditions["Positive News"]:
            reasons.append("Positive sentiment")
        
        return f"BUY - {len(reasons)} strong bullish factors: {', '.join(reasons)}"
    
    if strength_score >= 40:
        return "NEUTRAL - Mixed signals, wait for clearer trend"
    
    if strength_score >= 20:
        reasons = []
        if conditions["Bearish Pattern"]:
            reasons.append("Bearish patterns")
        if conditions["RSI Bullish"] is False:  # Changed from RSI < 30 to use the existing condition
            reasons.append("Overbought conditions")
        if conditions["Stochastic Bullish"] is False:
            reasons.append("Bearish divergence")
        
        return f"SELL - {len(reasons)} bearish factors: {', '.join(reasons)}"
    
    return "STRONG SELL - Strong bearish conditions aligned"

def get_nifty_200_stocks():
    """Return hardcoded list of Nifty 200 stocks"""
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
    print(nifty_200.__len__())
    # Sort alphabetically by company name
    sorted_stocks = dict(sorted(nifty_200.items()))
    return sorted_stocks

def analyze_stock(ticker):
    """Analyze a single stock with comprehensive technical and fundamental analysis"""
    try:
        print(f"Analyzing {ticker}...")
        
        # Download data with more history
        print("Downloading data...")

        df = yf.download(ticker, period="1y", interval="1d")       
        if df.empty:
            print(f"Error: No data found for {ticker}")
            return None
            
        print("Calculating technical indicators...")
        df = calculate_technical_indicators(df)
        if df is None:
            print("Error: Failed to calculate technical indicators")
            return None
            
        print("Calculating target and stop loss...")
        price_levels = calculate_target_and_sl(df)
        if price_levels is None:
            print("Error: Failed to calculate target and stop loss")
            return None
            
        print("Getting latest values...")
        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        ma20 = float(latest['MA20'])
        ma50 = float(latest['MA50'])
        ma200 = float(latest['MA200'])
        rsi = float(latest['RSI'])
        macd = float(latest['MACD'])
        macd_signal = float(latest['MACD_signal'])
        stoch_k = float(latest['Stoch_K'])
        stoch_d = float(latest['Stoch_D'])
        volume = float(latest['Volume'])
        volume_ma20 = float(latest['Volume_MA20'])
        
        print("Calculating additional metrics...")
        daily_change = ((current_price - float(df['Close'].iloc[-2])) / float(df['Close'].iloc[-2])) * 100
        volume_change = ((volume - float(df['Volume'].iloc[-2])) / float(df['Volume'].iloc[-2])) * 100
        
        print("Calculating trend directions...")
        short_term_trend = "Bullish" if current_price > ma20 else "Bearish"
        medium_term_trend = "Bullish" if current_price > ma50 else "Bearish"
        long_term_trend = "Bullish" if current_price > ma200 else "Bearish"
        
        print("Calculating volatility...")
        daily_returns = df['Close'].pct_change()
        annual_volatility = daily_returns.std() * np.sqrt(252) * 100
        volatility = annual_volatility.item() if not pd.isna(annual_volatility).any() else 0.0
        
        print("Calculating support and resistance levels...")
        support_levels = calculate_support_levels(df)
        resistance_levels = calculate_resistance_levels(df)
        
        # Ensure we have a list of floats for support levels
        if not support_levels:
            support_levels = []
        min_support = min(support_levels) if support_levels else 0
        
        print("Checking candlestick patterns...")
        patterns = check_candlestick_patterns(df)
        
        print("Getting news sentiment...")
        news_data = get_news_sentiment(ticker)
        
        print("Defining conditions...")
        conditions = {
            # Trend Conditions
            "Strong Uptrend": current_price > ma20 and current_price > ma50 and current_price > ma200,
            "Price > MA20": current_price > ma20,
            "Price > MA50": current_price > ma50,
            "Price > MA200": current_price > ma200,
            
            # Momentum Conditions
            "RSI Bullish": 40 <= rsi <= 70,
            "MACD Bullish": macd > macd_signal and macd > 0,
            "Stochastic Bullish": stoch_k > stoch_d and stoch_k <= 80,
            
            # Volume Conditions
            "Strong Volume": volume > volume_ma20 * 1.5,
            "Increasing Volume": volume_change > 20,
            
            # Pattern Conditions
            "Bullish Pattern": any(p in ["Hammer", "Morning Star", "Bullish Engulfing", "Bullish Harami"] for p in patterns),
            "Bearish Pattern": any(p in ["Evening Star", "Bearish Engulfing", "Bearish Harami", "Shooting Star"] for p in patterns),
            
            # Risk Management
            "Good Risk-Reward": price_levels['risk_reward_ratio'] >= 2.0,
            "Reasonable Volatility": 15 <= volatility <= 40,
            
            # Market Sentiment
            "Positive News": news_data["sentiment_label"] == "Positive",
            "Above Support": current_price > min_support if support_levels else False
        }
        
        print("Calculating strength score...")
        strength_score = (sum(conditions.values()) / len(conditions)) * 100
        
        print("Creating result dictionary...")
        result = {
            "ticker": ticker,
            "price": current_price,
            "daily_change": round(daily_change, 2),
            "target": price_levels['target_price'],
            "stop_loss": price_levels['stop_loss'],
            "risk_reward": price_levels['risk_reward_ratio'],
            "support_levels": support_levels[:3],
            "resistance_levels": resistance_levels[:3],
            "rsi": round(rsi, 2),
            "macd": round(macd, 4),
            "stoch_k": round(stoch_k, 2),
            "stoch_d": round(stoch_d, 2),
            "atr": price_levels['atr'],
            "volatility": round(volatility, 2),
            "volume": volume,
            "volume_change": round(volume_change, 2),
            "volume_ma20": volume_ma20,
            "short_term_trend": short_term_trend,
            "medium_term_trend": medium_term_trend,
            "long_term_trend": long_term_trend,
            "candlestick_patterns": patterns,
            "news_sentiment": news_data["sentiment_label"],
            "sentiment_score": news_data["sentiment_score"],
            "latest_news": news_data["latest_news"],
            "conditions": conditions,
            "conditions_met": sum(conditions.values()),
            "total_conditions": len(conditions),
            "strength_score": round(strength_score, 2),
            "analysis_summary": generate_analysis_summary(conditions, patterns, news_data),
            "trade_recommendation": generate_trade_recommendation(strength_score, conditions),
            "data": df
        }
        
        print(f"Analysis completed for {ticker}")
        return result
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
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
    
    # Add custom CSS for styling
    st.markdown("""
    <style>
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
                        st.markdown(f"""
                        <div class="pattern-card">
                            <p class="{'positive' if 'Bullish' in pattern else 'negative'}">
                                {pattern}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="pattern-card">
                        <p class="neutral">No significant patterns detected</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Analysis Summary
                st.markdown("<h2 class='section-title'>Analysis Summary</h2>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="summary-card">
                    <p>{result['analysis_summary']}</p>
                </div>
                """, unsafe_allow_html=True)
            
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
                                <span class="neutral">Source: {news.get('source', 'Unknown')}</span>
                                <span class="neutral">Date: {news.get('date', 'Unknown')}</span>
                                <span class="{'positive' if news.get('sentiment', '') == 'Positive' else 'negative'}">
                                    Sentiment: {news.get('sentiment', 'Unknown')}
                                </span>
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

if __name__ == "__main__":
    main()