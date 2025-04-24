import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import io # Import io for handling file-like objects

# --- Options Analysis Function ---

def analyze_options_only(options_data, underlying_price, target_expiry=None, message_list=None):
    """
    Analyzes NSE options chain data from a JSON dictionary (options-only version).

    This version focuses solely on options chain metrics.

    Args:
        options_data (dict): The parsed JSON data (dictionary) containing options chain.
                             Expected structure: {'records': {'data': [...], 'expiryDates': [...]}}
        underlying_price (float): The current price of the underlying asset.
        target_expiry (str, optional): The expiry date to filter for (e.g., "24-Apr-2025").
                                        Defaults to None (analyzes all expiries).
        message_list (list, optional): A list to append informational, warning,
                                       and error messages to. Each message is a dict
                                       like {'type': 'info', 'text': '...'}.
                                       Defaults to None (messages are not stored/returned).

    Returns:
        dict: A dictionary containing key options metrics and strike-wise data,
              or None if data is invalid or processing fails.
              Includes 'available_expiries' key even if no data is processed.
    """
    # Helper to append messages
    def add_message(type, text):
        if message_list is not None:
            message_list.append({'type': type, 'text': text})

    if not isinstance(options_data, dict) or 'records' not in options_data or 'data' not in options_data['records']:
        add_message('error', "Invalid input data structure. Expected a dictionary with 'records' -> 'data'.")
        return None

    raw_options_list = options_data['records']['data']
    available_expiries_from_json = options_data.get('records', {}).get('expiryDates')

    if not raw_options_list or not isinstance(raw_options_list, list):
        add_message('info', "Input data contains no options data in 'records.data'.")
        # Return available expiries even if no data is processed
        if available_expiries_from_json and isinstance(available_expiries_from_json, list):
             return {'available_expiries': sorted(available_expiries_from_json), 'filtered_expiry': None}
        return None # No data and no expiries list


    # --- Transform JSON data into Standardized Long Format DataFrame ---
    standardized_data = []

    for item in raw_options_list:
        # Each item might contain a 'CE' key, a 'PE' key, or both, plus common fields like strikePrice and expiryDate
        strike_price = item.get('strikePrice')
        expiry_date = item.get('expiryDate')
        ce_data = item.get('CE')
        pe_data = item.get('PE')

        # Skip if essential data is missing or invalid
        # Ensure strike_price is a number and expiry_date is a string
        if strike_price is None or not isinstance(strike_price, (int, float)) or expiry_date is None or not isinstance(expiry_date, str):
             add_message('warning', f"Skipping malformed data item: missing strikePrice, expiryDate, or invalid type. Item: {item}")
             continue # Skip malformed item


        # --- Define key mapping from JSON keys to standardized names ---
        # This mapping should cover all fields needed for analysis and display
        json_key_mapping = {
            'openInterest': 'Open Interest',
            'changeinOpenInterest': 'Change in OI',
            'totalTradedVolume': 'Volume',
            'impliedVolatility': 'Implied Volatility',
            'lastPrice': 'Last Traded Price',
            'change': 'Change', # Daily change in LTP
            'bidQty': 'Bid Quantity',
            'bidprice': 'Bid Price',
            'askQty': 'Ask Quantity',
            'askPrice': 'Ask Price',
            'underlyingValue': 'Underlying Value',
             'pchangeinOpenInterest': 'Pchange in OI', # Percentage change
             'totalBuyQuantity': 'Total Buy Quantity',
             'totalSellQuantity': 'Total Sell Quantity'
        }

        # Process CE data if available and is a dictionary
        if ce_data and isinstance(ce_data, dict):
             ce_entry = {
                 'Strike Price': strike_price,
                 'Expiry Date': expiry_date,
                 'Option Type': 'CE',
                 # Get mapped values, defaulting to None if key is missing
                 **{std_name: ce_data.get(json_key) for json_key, std_name in json_key_mapping.items()}
             }
             standardized_data.append(ce_entry)


        # Process PE data if available and is a dictionary
        if pe_data and isinstance(pe_data, dict):
             pe_entry = {
                 'Strike Price': strike_price,
                 'Expiry Date': expiry_date,
                 'Option Type': 'PE',
                 # Get mapped values, defaulting to None if key is missing
                 **{std_name: pe_data.get(json_key) for json_key, std_name in json_key_mapping.items()}
             }
             standardized_data.append(pe_entry)
   
    if not standardized_data:
        add_message('info', "No valid options contract data found in the JSON 'records.data' after initial processing.")
        # Return available expiries even if no data is processed
        if available_expiries_from_json and isinstance(available_expiries_from_json, list):
             return {'available_expiries': sorted(available_expiries_from_json), 'filtered_expiry': None}
        return None # No data and no expiries list


    options_df_standard = pd.DataFrame(standardized_data)    

    # --- Data Cleaning ---
    # Ensure essential columns are numeric, coercing errors
    numeric_cols = [
        'Strike Price', 'Open Interest', 'Change in OI', 'Volume',
        'Implied Volatility', 'Last Traded Price', 'Change',
        'Bid Quantity', 'Bid Price', 'Ask Quantity', 'Ask Price', 'Underlying Value',
        'Pchange in OI', 'Total Buy Quantity', 'Total Sell Quantity'
    ]

    for col in numeric_cols:
        if col in options_df_standard.columns:
             # Coerce errors to NaN - handle potential non-numeric values from get()
            options_df_standard[col] = pd.to_numeric(options_df_standard[col], errors='coerce')
        # else: # Column not found, ignore (e.g., Pchange in OI might not always be there)
             # add_message('warning', f"Column '{col}' not found in standardized options data.")


    # Drop rows where essential columns might be NaN after cleaning/conversion
    initial_rows = len(options_df_standard)
    options_df_standard.dropna(subset=['Strike Price', 'Open Interest', 'Volume', 'Option Type'], inplace=True)
    if len(options_df_standard) < initial_rows:
         add_message('info', f"Removed {initial_rows - len(options_df_standard)} rows with missing essential data after cleaning.")


    # --- Expiry Date Filtering ---
    available_expiries = None # Initialize
    # Ensure 'Expiry Date' column exists and is string type before parsing
    if 'Expiry Date' in options_df_standard.columns and pd.api.types.is_string_dtype(options_df_standard['Expiry Date']):
         # Get all unique expiry dates from the data
         try:
             # Convert Expiry Date column to datetime for filtering and consistent format
             options_df_standard['Expiry Date_dt'] = pd.to_datetime(options_df_standard['Expiry Date'], format='%d-%b-%Y', errors='coerce')
             # Drop rows where date conversion failed
             initial_rows = len(options_df_standard)
             options_df_standard.dropna(subset=['Expiry Date_dt'], inplace=True)
             if len(options_df_standard) < initial_rows:
                  add_message('info', f"Removed {initial_rows - len(options_df_standard)} rows with invalid expiry date format.")

             # Get unique expiry dates as strings in the original format
             available_expiries = sorted(options_df_standard['Expiry Date_dt'].dt.strftime('%d-%b-%Y').unique().tolist())

         except Exception as e:
             add_message('warning', f"Could not parse Expiry Date column: {e}. Expiry filtering disabled.")
             options_df_standard['Expiry Date_dt'] = pd.NaT # Ensure datetime column exists even if parsing fails
             available_expiries = None # Cannot filter if dates aren't parsed
    else:
         add_message('warning', "Expiry Date column not found or is not string type. Expiry filtering disabled.")
         options_df_standard['Expiry Date_dt'] = pd.NaT
         available_expiries = None


    filtered_expiry_display = target_expiry # Store the requested expiry for display


    # If a target_expiry is provided AND we have successfully parsed dates, filter the data
    if target_expiry and available_expiries is not None and target_expiry in available_expiries and 'Expiry Date_dt' in options_df_standard.columns and not options_df_standard['Expiry Date_dt'].empty:
         try:
             target_dt = pd.to_datetime(target_expiry, format='%d-%b-%Y')
             initial_rows = len(options_df_standard)
             options_df_standard = options_df_standard[options_df_standard['Expiry Date_dt'] == target_dt].copy()            
             
             # If filtering results in empty DataFrame, show info message
             if options_df_standard.empty:
                  add_message('info', f"No options data found for expiry date: {target_expiry}")
         except Exception as e:
             add_message('warning', f"Error filtering by expiry date '{target_expiry}': {e}. Analyzing all expiries.")
             filtered_expiry_display = "Error filtering - All expiries analyzed" # Update display status
             # options_df_standard remains unfiltered


    if options_df_standard.empty:
        # Pass available expiries even if the final filtered DF is empty
        return {'available_expiries': available_expiries, 'filtered_expiry': filtered_expiry_display}
        # Return only expiries if no data is available after filtering/cleaning


    # --- Now use this options_df_standard for calculations ---
    calls = options_df_standard[options_df_standard['Option Type'] == 'CE'].copy()
    puts = options_df_standard[options_df_standard['Option Type'] == 'PE'].copy()

    # Ensure sums are calculated only on valid numeric data
    total_calls_oi = calls['Open Interest'].sum() if not calls.empty and 'Open Interest' in calls.columns else 0
    total_puts_oi = puts['Open Interest'].sum() if not puts.empty and 'Open Interest' in puts.columns else 0
    total_calls_volume = calls['Volume'].sum() if not calls.empty and 'Volume' in calls.columns else 0
    total_puts_volume = puts['Volume'].sum() if not puts.empty and 'Volume' in puts.columns else 0


    # Put/Call Ratio (PCR) based on OI
    pcr_oi = total_puts_oi / total_calls_oi if total_calls_oi > 0 else np.inf # Use inf for division by zero

    # Put/Call Ratio (PCR) based on Volume
    pcr_volume = total_puts_volume / total_calls_volume if total_calls_volume > 0 else np.inf # Use inf


    # Max Pain calculation
    strikes = options_df_standard['Strike Price'].unique()
    total_losses = {}

   # Use the standardized DFs for merging. Ensure 'Open Interest' exists before merging.
    if 'Open Interest' in calls.columns and 'Open Interest' in puts.columns and 'Strike Price' in calls.columns and 'Strike Price' in puts.columns:
         # For simplicity, calculate Max Pain only on the CURRENTLY FILTERED (or unfiltered) data
         merged_strikes = pd.merge(calls[['Strike Price', 'Open Interest']].rename(columns={'Open Interest': 'Call_OI'}),
                                   puts[['Strike Price', 'Open Interest']].rename(columns={'Open Interest': 'Put_OI'}),
                                   on='Strike Price', how='outer').fillna(0)

         # Ensure OI columns are numeric before calculation (Strike Price should already be numeric from earlier cleaning)
         merged_strikes['Call_OI'] = pd.to_numeric(merged_strikes['Call_OI'], errors='coerce').fillna(0)
         merged_strikes['Put_OI'] = pd.to_numeric(merged_strikes['Put_OI'], errors='coerce').fillna(0)

         # --- Set Strike Price as index ---
         # 'Strike Price' is already numeric from earlier cleaning and used in the merge
         merged_strikes.set_index('Strike Price', inplace=True)
         # -------------------------------------------------

    else:
         add_message('warning', "Required columns for Max Pain calculation not found after standardizing options data.")
         # Create an empty DataFrame with expected structure if columns are missing
         merged_strikes = pd.DataFrame(columns=['Strike Price', 'Call_OI', 'Put_OI'])


    # Max Pain calculation - only calculate if underlying_price is valid and there are strikes
    max_pain_strike = None
    # Check if current_price is a valid number before using it and if merged_strikes is not emptyt
    if isinstance(underlying_price, (float, int)) and not pd.isna(underlying_price) and underlying_price > 0 and not merged_strikes.empty:

        # Calculate total loss at each strike across the available data
        # 'Strike Price' is now the INDEX of merged_strikes. Access the index value using row.name
        total_losses_at_strikes = merged_strikes.apply(
            lambda row: max(0, row.name - underlying_price) * row['Call_OI'] + max(0, underlying_price - row.name) * row['Put_OI'],
            axis=1 # Apply function row-wise
        )

        # Max pain is the strike with the minimum total loss
        if not total_losses_at_strikes.empty and not total_losses_at_strikes.isna().all():
            # Find the index (strike price) with the minimum value
            # idxmin() will now return the Strike Price because it's the index label
            max_pain_strike = total_losses_at_strikes.idxmin()

        else:
             add_message('warning', "No valid strikes found for Max Pain calculation after cleaning or calculation resulted in NaNs.")
    else:
         # Handle cases where underlying price is invalid or data is empty after filtering
         if isinstance(underlying_price, (float, int)) and not pd.isna(underlying_price) and underlying_price <= 0:
              add_message('warning', f"Max Pain calculation skipped due to invalid underlying price: {underlying_price}.")
         elif merged_strikes.empty:
               add_message('info', "Max Pain calculation skipped as options data is empty after processing.")

    # Strike-wise OI, Change in OI, Volume, IV, LTP (aggregated across expiries if not filtered)
    # Group by Strike Price and Option Type to aggregate data for the strike-wise table
    # Use the standardized options_df_standard directly
    # Aggregate by sum for OI, COI, Volume. For IV and LTP, mean makes sense if multiple expiries per strike.
    # If filtered by expiry, sum/mean/last will likely be the same for each strike/type combination.

    required_pivot_cols = ['Strike Price', 'Option Type', 'Open Interest', 'Change in OI', 'Volume', 'Implied Volatility', 'Last Traded Price']
    if all(col in options_df_standard.columns for col in required_pivot_cols):
         strike_analysis_df_standard = options_df_standard.pivot_table(
             index='Strike Price',
             columns='Option Type',
             values=['Open Interest', 'Change in OI', 'Volume', 'Implied Volatility', 'Last Traded Price'],
             aggfunc={'Open Interest': 'sum', 'Change in OI': 'sum', 'Volume': 'sum', 'Implied Volatility': 'mean', 'Last Traded Price': 'last'} # Use mean for IV
         ).fillna(0)

         # Flatten the multi-level columns created by pivot_table
         if isinstance(strike_analysis_df_standard.columns, pd.MultiIndex):
              strike_analysis_df_standard.columns = ['_'.join(map(str, col)).strip() for col in strike_analysis_df_standard.columns.values]

         # Rename to a more readable format
         strike_analysis_df_standard.rename(columns={
             'Open Interest_CE': 'CE_OI', 'Change in OI_CE': 'CE_Change in OI', 'Volume_CE': 'CE_Volume', 'Implied Volatility_CE': 'CE_IV', 'Last Traded Price_CE': 'CE_LTP',
             'Open Interest_PE': 'PE_OI', 'Change in OI_PE': 'PE_Change in OI', 'Volume_PE': 'PE_Volume', 'Implied Volatility_PE': 'PE_IV', 'Last Traded Price_PE': 'PE_LTP',
         }, inplace=True)

         # Add Total OI and Total Change in OI per strike
         strike_analysis_df_standard['Total_OI'] = strike_analysis_df_standard.get('CE_OI', 0) + strike_analysis_df_standard.get('PE_OI', 0)
         strike_analysis_df_standard['Total_Change_in_OI'] = strike_analysis_df_standard.get('CE_Change in OI', 0) + strike_analysis_df_standard.get('PE_Change in OI', 0)


         # Sort by Strike Price (index)
         strike_analysis_df_standard = strike_analysis_df_standard.sort_index()

    else:
        add_message('warning', "Required columns for strike-wise analysis not found after standardizing options data.")
        strike_analysis_df_standard = pd.DataFrame() # Empty DataFrame


    # --- Identify key strikes for support/resistance based on high OI ---
    # Find strikes with significantly high Call OI (potential resistance) and Put OI (potential support)
    call_resistance_strikes = []
    put_support_strikes = []

    if not strike_analysis_df_standard.empty:
        # Ensure columns exist and are not all NaN before calculating quantile
        if 'CE_OI' in strike_analysis_df_standard.columns and not strike_analysis_df_standard['CE_OI'].isna().all():
             # Calculate quantile only if there are non-zero OI values
             if (strike_analysis_df_standard['CE_OI'] > 0).sum() > 0:
                 oi_threshold_calls = strike_analysis_df_standard['CE_OI'].quantile(0.90) # Top 10% OI strikes
                 # Ensure the index (Strike Price) is numeric before comparison
                 potential_strikes = strike_analysis_df_standard[strike_analysis_df_standard['CE_OI'] > oi_threshold_calls].index
                 call_resistance_strikes = [s for s in potential_strikes if isinstance(s, (float, int)) and not pd.isna(s)] # Filter out non-numeric/NaN index values

        if 'PE_OI' in strike_analysis_df_standard.columns and not strike_analysis_df_standard['PE_OI'].isna().all():
             # Calculate quantile only if there are non-zero OI values
             if (strike_analysis_df_standard['PE_OI'] > 0).sum() > 0:
                oi_threshold_puts = strike_analysis_df_standard['PE_OI'].quantile(0.90)
                potential_strikes = strike_analysis_df_standard[strike_analysis_df_standard['PE_OI'] > oi_threshold_puts].index
                put_support_strikes = [s for s in potential_strikes if isinstance(s, (float, int)) and not pd.isna(s)] # Filter out non-numeric/NaN index values


    # Filter to strikes reasonably close to the current price
    # Only filter if current_price is valid and numeric
    if isinstance(underlying_price, (float, int)) and not pd.isna(underlying_price) and underlying_price > 0:
         price_range_factor = 0.10 # Consider strikes within 10% of current price
         call_resistance_strikes = [s for s in call_resistance_strikes if s >= underlying_price and s <= underlying_price * (1 + price_range_factor)]
         put_support_strikes = [s for s in put_support_strikes if s <= underlying_price and s >= underlying_price * (1 - price_range_factor)]
    else:
         # If current price is not available or invalid, return all significant OI strikes found globally
         pass # Lists already contain all significant strikes from quantile


    # Calculate Average IV across all options (or near ITM/ATM options)
    average_iv = np.nan # Default to NaN
    if not options_df_standard.empty and 'Implied Volatility' in options_df_standard.columns:
        # Calculate average IV only for strikes near the money (e.g., within 5% of price)
        # Or simply the average of all non-NaN, non-zero IVs
        valid_ivs = options_df_standard['Implied Volatility'].replace(0, np.nan).dropna() # Exclude 0 IVs
        if not valid_ivs.empty:
             average_iv = valid_ivs.mean()

    # Return the analysis results dictionary
    return {
        'total_calls_oi': int(total_calls_oi) if not pd.isna(total_calls_oi) else 0,
        'total_puts_oi': int(total_puts_oi) if not pd.isna(total_puts_oi) else 0,
        'pcr_oi': round(pcr_oi, 2) if isinstance(pcr_oi, (int, float)) and not np.isinf(pcr_oi) and not pd.isna(pcr_oi) else "N/A", # Use np.isinf
        'pcr_volume': round(pcr_volume, 2) if isinstance(pcr_volume, (int, float)) and not np.isinf(pcr_volume) and not pd.isna(pcr_volume) else "N/A", # Use np.isinf
        'max_pain_strike': int(max_pain_strike) if max_pain_strike is not None and isinstance(max_pain_strike, (float, int)) and not pd.isna(max_pain_strike) else "N/A",
        'strike_analysis_df': strike_analysis_df_standard,
        'call_resistance_strikes': sorted([int(s) for s in call_resistance_strikes]) if call_resistance_strikes else [],
        'put_support_strikes': sorted([int(s) for s in put_support_strikes]) if put_support_strikes else [],
        'average_iv': round(average_iv, 2) if isinstance(average_iv, (int, float)) and not pd.isna(average_iv) else "N/A",
        'available_expiries': available_expiries, # Include available expiries in the result
        'filtered_expiry': filtered_expiry_display # Include the filtered expiry status
    }


# --- Options-Based Recommendation Function ---

def generate_options_based_recommendation(options_analysis_results, underlying_price):
    """
    Generates a trade recommendation, target, SL, and success % based ONLY on options data.

    Args:
        options_analysis_results (dict): The result dictionary from analyze_options_only.
        underlying_price (float): The current price of the underlying asset.

    Returns:
        dict: Contains 'recommendation', 'target', 'stop_loss', 'success_percentage'.
    """
    recommendation = "NEUTRAL"
    target = "N/A"
    stop_loss = "N/A"
    success_percentage = 50 # Start with neutral 50%

    if options_analysis_results is None:
        return {
            'recommendation': "NEUTRAL - No options data available",
            'target': "N/A",
            'stop_loss': "N/A",
            'success_percentage': 50
        }

    pcr_oi = options_analysis_results.get('pcr_oi', 'N/A')
    max_pain_strike = options_analysis_results.get('max_pain_strike', 'N/A')
    call_resistance_strikes = options_analysis_results.get('call_resistance_strikes', [])
    put_support_strikes = options_analysis_results.get('put_support_strikes', [])
    average_iv = options_analysis_results.get('average_iv', 'N/A')

    # --- Determine Bias and Recommendation ---
    bias_score = 0 # Positive for bullish, negative for bearish

    # PCR Bias
    if isinstance(pcr_oi, (int, float)) and not np.isinf(pcr_oi):
        if pcr_oi < 0.8: bias_score += 2 # Very Bullish PCR
        elif pcr_oi < 1.0: bias_score += 1 # Bullish PCR
        elif pcr_oi > 1.2: bias_score -= 2 # Very Bearish PCR
        elif pcr_oi > 1.0: bias_score -= 1 # Bearish PCR

    # Max Pain vs. Current Price
    if isinstance(max_pain_strike, (int, float)) and isinstance(underlying_price, (int, float)):
        if max_pain_strike > underlying_price * 1.01: bias_score += 1 # Max Pain slightly above price
        elif max_pain_strike < underlying_price * 0.99: bias_score -= 1 # Max Pain slightly below price

    # High OI Strikes
    # Check if current price is above a significant Put OI strike
    if isinstance(underlying_price, (int, float)) and put_support_strikes:
        # Find the highest put support strike below or near current price
        highest_put_support_near_price = max([s for s in put_support_strikes if s <= underlying_price * 1.02], default=None)
        if highest_put_support_near_price is not None:
             bias_score += 1 # Price is above a key support strike

    # Check if current price is below a significant Call OI strike
    if isinstance(underlying_price, (int, float)) and call_resistance_strikes:
        # Find the lowest call resistance strike above or near current price
        lowest_call_resistance_near_price = min([s for s in call_resistance_strikes if s >= underlying_price * 0.98], default=None)
        if lowest_call_resistance_near_price is not None:
             bias_score -= 1 # Price is below a key resistance strike


    # IV Analysis (Simplified)
    if isinstance(average_iv, (int, float)):
        if average_iv > 50: # High IV could imply expected big moves or uncertainty
             pass # Don't add bias, but note high volatility

    # --- Determine Recommendation and Success Percentage based on Bias Score ---
    if bias_score >= 3:
        recommendation = "STRONG BUY"
        success_percentage = 80 # Higher confidence
    elif bias_score >= 1:
        recommendation = "BUY"
        success_percentage = 65 # Moderate confidence
    elif bias_score <= -3:
        recommendation = "STRONG SELL"
        success_percentage = 80 # Higher confidence
    elif bias_score <= -1:
        recommendation = "SELL"
        success_percentage = 65 # Moderate confidence
    else:
        recommendation = "NEUTRAL"
        success_percentage = 50 # Neutral confidence

    # --- Determine Target and Stop Loss (Options-Based) ---
    # This is a simplified approach using Max Pain and nearest high OI strikes
    if isinstance(underlying_price, (int, float)):
        if recommendation in ["BUY", "STRONG BUY"]:
            # Target: Nearest Call Resistance Strike above price, or Max Pain if higher
            potential_targets = [s for s in call_resistance_strikes if s > underlying_price]
            if isinstance(max_pain_strike, (int, float)) and max_pain_strike > underlying_price:
                 potential_targets.append(max_pain_strike)

            target = min(potential_targets, default=underlying_price * 1.05) # Default target if none found

            # Stop Loss: Nearest Put Support Strike below price, or a percentage below price
            potential_sls = [s for s in put_support_strikes if s < underlying_price]
            stop_loss = max(potential_sls, default=underlying_price * 0.95) # Default SL if none found

        elif recommendation in ["SELL", "STRONG SELL"]:
             # Target: Nearest Put Support Strike below price, or Max Pain if lower
            potential_targets = [s for s in put_support_strikes if s < underlying_price]
            if isinstance(max_pain_strike, (int, float)) and max_pain_strike < underlying_price:
                 potential_targets.append(max_pain_strike)

            target = max(potential_targets, default=underlying_price * 0.95) # Default target if none found

            # Stop Loss: Nearest Call Resistance Strike above price, or a percentage above price
            potential_sls = [s for s in call_resistance_strikes if s > underlying_price]
            stop_loss = min(potential_sls, default=underlying_price * 1.05) # Default SL if none found

        else: # Neutral
            # Target: Nearest resistance strike above price
            potential_targets = [s for s in call_resistance_strikes if s > underlying_price]
            target = min(potential_targets, default="N/A")
            # Stop Loss: Nearest support strike below price
            potential_sls = [s for s in put_support_strikes if s < underlying_price]
            stop_loss = max(potential_sls, default="N/A")

    # Ensure target and stop loss are distinct and sensible relative to current price
    if isinstance(target, (int, float)) and isinstance(stop_loss, (int, float)) and isinstance(underlying_price, (int, float)):
        if recommendation in ["BUY", "STRONG BUY"]:
            if target <= underlying_price: target = underlying_price * 1.01 # Ensure target is above price
            if stop_loss >= underlying_price: stop_loss = underlying_price * 0.99 # Ensure SL is below price
            if stop_loss <= 0: stop_loss = underlying_price * 0.5 # Ensure positive SL

        elif recommendation in ["SELL", "STRONG SELL"]:
            if target >= underlying_price: target = underlying_price * 0.99 # Ensure target is below price
            if stop_loss <= underlying_price: stop_loss = underlying_price * 1.01 # Ensure SL is above price
             # No check for positive SL needed for sell side, but target should be positive
            if target <= 0: target = underlying_price * 0.5


    # Round numeric results
    target = round(target, 2) if isinstance(target, (int, float)) else target
    stop_loss = round(stop_loss, 2) if isinstance(stop_loss, (int, float)) else stop_loss


    return {
        'recommendation': recommendation,
        'target': target,
        'stop_loss': stop_loss,
        'success_percentage': success_percentage
    }


# --- Streamlit App Main Function ---

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Options Chain Analyzer",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add custom CSS for styling (Simplified for this options-only version)
    st.markdown("""
    <style>
    .stAppToolbar { display: none; }
    .metric-card { background: linear-gradient(135deg, #f0f2f6 0%, #e0e2e6 100%); padding: 1.5rem; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 1rem; transition: transform 0.3s ease; }
    .metric-card:hover { transform: translateY(-5px); }

    .positive { color: #22c55e; }
    .negative { color: #ef4444; }
    .neutral { color: #6b7280; }

    .pcr-bullish { color: #22c55e; font-weight: bold; }
    .pcr-bearish { color: #ef4444; font-weight: bold; }
    .pcr-neutral { color: #6b7280; }

    .recommendation-box { background: linear-gradient(135deg, #1e90ff 0%, #22c55e 100%); color: white; padding: 1.5rem; border-radius: 1rem; text-align: center; font-weight: bold; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2); }
    .recommendation-box.SELL { background: linear-gradient(135deg, #ef4444 0%, #f87171 100%); } /* Red gradient for Sell */
    .recommendation-box.NEUTRAL { background: linear-gradient(135deg, #6b7280 0%, #9ca3af 100%); } /* Gray gradient for Neutral */


    .strength-meter { height: 10px; border-radius: 5px; background: #e5e7eb; overflow: hidden; margin-top: 10px;}
    .strength-meter-fill { height: 100%; background: linear-gradient(90deg, #1e90ff 0%, #22c55e 100%); transition: width 0.5s ease; }

    .section-title { font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem; color: #1e90ff; }
    .section-title h3 { color: #1e90ff; margin-top: 0; } /* Style for H3 within cards */

    /* Footer styling */
    .footer {
        margin-top: 50px;
        text-align: center;
        font-size: 0.9rem;
        color: #6b7280;
    }

    /* Adjust Streamlit default margins */
     .css-18e3th9 { # main container
         padding-top: 1rem;
         padding-right: 1rem;
         padding-left: 1rem;
         padding-bottom: 1rem;
     }
    .css-1d3z3hb { # sidebar
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Options Chain Analyzer")
    st.write("Upload NSE Options Chain JSON data to get analysis based on options metrics.")

    # --- Sidebar Inputs ---
    st.sidebar.header("Input Data")

    # Input for current underlying price
    current_price_input = st.sidebar.number_input(
        "Enter Current Underlying Price",
        min_value=0.01,
        format="%.2f",
        key='current_price_input'
    )

    # Options Chain Upload
    st.sidebar.markdown("---")
    st.sidebar.header("Upload Options Chain Data (JSON)")

    # Initialize state variables for file upload and expiries
    if 'options_file_only' not in st.session_state:
         st.session_state['options_file_only'] = None
    if 'available_expiries_only' not in st.session_state:
         st.session_state['available_expiries_only'] = None
    if 'uploaded_options_filename_only' not in st.session_state:
         st.session_state['uploaded_options_filename_only'] = None
    # Placeholder for file upload status messages
    file_upload_status_placeholder = st.sidebar.empty()


    options_file_uploader = st.sidebar.file_uploader(
        "Upload NSE Options Chain JSON",
        type=['json'],
        key='options_uploader_json_only', # Unique key for this app
        help="Upload the Options Chain data JSON file."
    )

    # --- Handle file upload and extract expiries ---
    # Check if a new file was uploaded (widget value is not None and filename changed)
    if options_file_uploader is not None and st.session_state.get('uploaded_options_filename_only') != options_file_uploader.name:
        st.session_state['options_file_only'] = options_file_uploader
        st.session_state['uploaded_options_filename_only'] = options_file_uploader.name # Store filename
        st.session_state['available_expiries_only'] = None # Clear previous expiries
        # Clear any previous analysis results and messages
        if 'options_analysis_results_only' in st.session_state: del st.session_state['options_analysis_results_only']
        if 'options_analysis_messages_only' in st.session_state: del st.session_state['options_analysis_messages_only']


        # Attempt to read JSON and get expiry dates immediately after upload
        try:
            # Need to reset file pointer after reading for expiry dates
            options_file_uploader.seek(0)
            temp_json_data = json.load(options_file_uploader)
            expiries_list = temp_json_data.get('records', {}).get('expiryDates')
            if expiries_list and isinstance(expiries_list, list):
                st.session_state['available_expiries_only'] = expiries_list
                file_upload_status_placeholder.success(f"Options JSON uploaded successfully. Found {len(expiries_list)} expiries.")
            else:
                 st.session_state['available_expiries_only'] = ["No expiries found in JSON"]
                 file_upload_status_placeholder.warning("Options JSON uploaded, but could not find expiry dates in 'records.expiryDates'.")

            # Reset file pointer again for the actual analysis
            options_file_uploader.seek(0)

        except json.JSONDecodeError:
            file_upload_status_placeholder.error("Invalid JSON format in the uploaded file.")
            st.session_state['options_file_only'] = None
            st.session_state['available_expiries_only'] = None
            st.session_state['uploaded_options_filename_only'] = None
        except Exception as e:
            file_upload_status_placeholder.error(f"Error reading JSON for expiries: {str(e)}")
            st.session_state['options_file_only'] = None
            st.session_state['available_expiries_only'] = None
            st.session_state['uploaded_options_filename_only'] = None

    elif options_file_uploader is None and st.session_state.get('options_file_only') is not None:
         # If the uploader was previously filled but is now cleared by the user
         file_upload_status_placeholder.info("Options JSON file cleared.")
         st.session_state['options_file_only'] = None
         st.session_state['available_expiries_only'] = None
         st.session_state['uploaded_options_filename_only'] = None
         # Clear previous analysis results and messages when file is cleared
         if 'options_analysis_results_only' in st.session_state: del st.session_state['options_analysis_results_only']
         if 'options_analysis_messages_only' in st.session_state: del st.session_state['options_analysis_messages_only']
         # Rerun to update the display
         st.rerun()
    elif st.session_state.get('options_file_only') is not None:
         # If the same file is re-uploaded or app reruns with file in state,
         # display the previous status message from state
         if st.session_state.get('available_expiries_only') and st.session_state['available_expiries_only'] != ["No expiries found in JSON"]:
             file_upload_status_placeholder.success(f"Using uploaded Options JSON. Found {len(st.session_state['available_expiries_only'])} expiries.")
         elif st.session_state.get('uploaded_options_filename_only'):
              file_upload_status_placeholder.warning(f"Using uploaded Options JSON. Could not find expiry dates.")
         # Any error during initial upload read won't be re-displayed here, relying on the uploader state.


    # --- Expiry Date Selection ---
    selected_expiry = None
    # Only show expiry selection if expiries were successfully extracted and it's not the "No expiries" placeholder
    if st.session_state.get('available_expiries_only') and isinstance(st.session_state['available_expiries_only'], list) and len(st.session_state['available_expiries_only']) > 0 and st.session_state['available_expiries_only'] != ["No expiries found in JSON"]:
        st.sidebar.markdown("---")
        st.sidebar.header("Filter Options Data")
        selected_expiry = st.sidebar.selectbox(
            "Select Expiry Date (Optional)",
            options=["Analyze all expiries"] + st.session_state['available_expiries_only'],
            index=0, # Default to "Analyze all expiries"
            key='selected_expiry_selectbox_only' # Unique key
        )
        if selected_expiry == "Analyze all expiries":
             selected_expiry = None # Pass None to analyze_options_only


    # --- Analyze Button ---
    st.sidebar.markdown("---")
    if st.sidebar.button("Analyze Options Data", key='analyze_options_button_only', use_container_width=True):
        # Clear previous analysis results and messages when starting a new analysis
        if 'options_analysis_results_only' in st.session_state: del st.session_state['options_analysis_results_only']
        st.session_state['options_analysis_messages_only'] = [] # Clear messages list

        if current_price_input <= 0:
            st.session_state['options_analysis_messages_only'].append({'type': 'error', 'text': "Please enter a valid current underlying price."})
            st.rerun()
            return

        options_file_for_analysis = st.session_state.get('options_file_only')

        if options_file_for_analysis is None:
            st.session_state['options_analysis_messages_only'].append({'type': 'warning', 'text': "Please upload an Options Chain JSON file."})
            st.rerun()
            return

        # Read the JSON data from the uploaded file
        try:
            options_file_for_analysis.seek(0) # Ensure file pointer is at the beginning
            options_data = json.load(options_file_for_analysis)
        except json.JSONDecodeError:
            st.session_state['options_analysis_messages_only'].append({'type': 'error', 'text': "Invalid JSON format in the uploaded file."})
            st.session_state['options_file_only'] = None # Clear file state on error
            st.session_state['available_expiries_only'] = None
            st.session_state['uploaded_options_filename_only'] = None
            st.rerun()
            return
        except Exception as e:
            st.session_state['options_analysis_messages_only'].append({'type': 'error', 'text': f"Error reading uploaded JSON file: {str(e)}"})
            st.session_state['options_file_only'] = None # Clear file state on error
            st.session_state['available_expiries_only'] = None
            st.session_state['uploaded_options_filename_only'] = None
            st.rerun()
            return


        # Perform options analysis
        with st.spinner("Analyzing options data..."):
            options_results = analyze_options_only(
                options_data,
                current_price_input,
                target_expiry=selected_expiry,
                message_list=st.session_state['options_analysis_messages_only'] # Pass the list from state
            )

        # Store the result in session state
        st.session_state['options_analysis_results_only'] = options_results

        # Force a rerun to display the results and messages
        st.rerun()


    # --- Display Messages from Analysis ---
    # Display any stored messages at the top of the main content area
    if 'options_analysis_messages_only' in st.session_state and st.session_state['options_analysis_messages_only']:
         for message in st.session_state['options_analysis_messages_only']:
             if message['type'] == 'error':
                  st.error(message['text'])
             elif message['type'] == 'warning':
                  st.warning(message['text'])
             elif message['type'] == 'info':
                  st.info(message['text'])
         # Keep messages until a new analysis is run or file is cleared


    # --- Display Results (Only if analysis result is in session state) ---
    if 'options_analysis_results_only' in st.session_state and st.session_state['options_analysis_results_only'] is not None:
        options_results = st.session_state['options_analysis_results_only']

        # Check if the result contains actual analysis data or just expiry info
        if 'total_calls_oi' in options_results: # Check for a key that indicates successful data processing

             st.markdown("<h2 class='section-title'>Options Analysis Results</h2>", unsafe_allow_html=True)

             # Display the expiry date that was used for filtering (or "All expiries")
             filtered_expiry_display = options_results.get('filtered_expiry')
             available_expiries = options_results.get('available_expiries')

             expiry_status_text = "Analyzing "
             if filtered_expiry_display:
                  expiry_status_text += f"data filtered for expiry date: **{filtered_expiry_display}**"
             elif available_expiries and available_expiries != ["No expiries found in JSON"]:
                   expiry_status_text += f"data across all expiries ({', '.join(available_expiries)})."
             elif st.session_state.get('uploaded_options_filename_only'): # File uploaded but no expiries found
                  expiry_status_text += "uploaded data (expiry information not found)."
             else: # Should not happen if options_results is not None and has data, but as a fallback
                  expiry_status_text = "Options data processed."

             st.info(expiry_status_text)


             # Display key metrics in metric cards
             st.markdown("<h3>Key Options Metrics</h3>", unsafe_allow_html=True)
             col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)

             with col_opt1:
                 total_calls_oi_val = options_results.get('total_calls_oi', 'N/A')
                 display_calls_oi = f"{total_calls_oi_val:,}" if isinstance(total_calls_oi_val, (int, float)) else total_calls_oi_val
                 st.markdown(f"""
                 <div class="metric-card">
                     <h3>Total Calls OI</h3>
                     <p>{display_calls_oi}</p>
                 </div>
                 """, unsafe_allow_html=True)
             with col_opt2:
                 total_puts_oi_val = options_results.get('total_puts_oi', 'N/A')
                 display_puts_oi = f"{total_puts_oi_val:,}" if isinstance(total_puts_oi_val, (int, float)) else total_puts_oi_val
                 st.markdown(f"""
                 <div class="metric-card">
                     <h3>Total Puts OI</h3>
                     <p>{display_puts_oi}</p>
                 </div>
                 """, unsafe_allow_html=True)
             with col_opt3:
                 pcr_oi = options_results.get('pcr_oi', 'N/A')
                 pcr_color = 'pcr-neutral'
                 if isinstance(pcr_oi, (int, float)) and not np.isinf(pcr_oi):
                     if pcr_oi < 1.0: pcr_color = 'pcr-bullish'
                     elif pcr_oi > 1.0: pcr_color = 'pcr-bearish'

                 st.markdown(f"""
                 <div class="metric-card">
                     <h3>PCR (OI)</h3>
                     <p class="{pcr_color}">{pcr_oi}</p>
                 </div>
                 """, unsafe_allow_html=True)
             with col_opt4:
                 max_pain = options_results.get('max_pain_strike', 'N/A')
                 display_max_pain = f"₹{max_pain:,}" if isinstance(max_pain, (int, float)) else max_pain
                 st.markdown(f"""
                 <div class="metric-card">
                     <h3>Max Pain</h3>
                     <p>{display_max_pain}</p>
                 </div>
                 """, unsafe_allow_html=True)

             # Add Average IV
             col_iv, _, _, _ = st.columns(4)
             with col_iv:
                 avg_iv = options_results.get('average_iv', 'N/A')
                 st.markdown(f"""
                 <div class="metric-card">
                     <h3>Average IV</h3>
                     <p>{avg_iv}{'%' if isinstance(avg_iv, (int, float)) else ''}</p>
                 </div>
                 """, unsafe_allow_html=True)


             st.markdown("<br>", unsafe_allow_html=True)

             # Display Strike-wise data table
             st.markdown("<h3>Strike-wise Data Overview</h3>", unsafe_allow_html=True)
             strike_analysis_df = options_results.get('strike_analysis_df')
             if strike_analysis_df is not None and not strike_analysis_df.empty:
                display_cols = ['CE_OI', 'CE_Change in OI', 'CE_Volume', 'CE_IV', 'CE_LTP',
                                'PE_OI', 'PE_Change in OI', 'PE_Volume', 'PE_IV', 'PE_LTP',
                                'Total_OI', 'Total_Change_in_OI']
                display_cols_existing = [col for col in strike_analysis_df.columns if col in display_cols]
                display_cols_ordered = [col for col in display_cols if col in display_cols_existing]

                st.dataframe(strike_analysis_df[display_cols_ordered].round(2), use_container_width=True)
             else:
                st.info("No strike-wise data available.")


             st.markdown("<h3>Key OI Levels (Potential S/R)</h3>", unsafe_allow_html=True)
             col_oi_levels1, col_oi_levels2 = st.columns(2)

             with col_oi_levels1:
                   call_strikes = options_results.get('call_resistance_strikes', [])
                   st.markdown(f"""
                   <div class="metric-card">
                       <div class="section-title">Potential Resistance (High Call OI)</div>
                       <p>{', '.join(map(lambda x: f"₹{x:,}", call_strikes)) if call_strikes else 'None identified near price'}</p>
                   </div>
                   """, unsafe_allow_html=True)
             with col_oi_levels2:
                 put_strikes = options_results.get('put_support_strikes', [])
                 st.markdown(f"""
                 <div class="metric-card">
                     <div class="section-title">Potential Support (High Put OI)</div>
                     <p>{', '.join(map(lambda x: f"₹{x:,}", put_strikes)) if put_strikes else 'None identified near price'}</p>
                  </div>
                  """, unsafe_allow_html=True)


             # --- Options-Based Recommendation ---
             st.markdown("<h2 class='section-title'>Options-Based Recommendation</h2>", unsafe_allow_html=True)
             # Pass options analysis results and current price to the recommendation function
             recommendation_results = generate_options_based_recommendation(options_results, current_price_input)

             recommendation = recommendation_results['recommendation']
             target = recommendation_results['target']
             stop_loss = recommendation_results['stop_loss']
             success_percentage = recommendation_results['success_percentage']

             # Determine color class for recommendation box
             rec_class = ""
             if "BUY" in recommendation: rec_class = "BUY"
             elif "SELL" in recommendation: rec_class = "SELL"
             else: rec_class = "NEUTRAL"


             st.markdown(f"""
             <div class="recommendation-box {rec_class}">
                 <h2>{recommendation}</h2>
                 <p>Target: {f'₹{target:,}' if isinstance(target, (int, float)) else target}</p>
                 <p>Stop Loss: {f'₹{stop_loss:,}' if isinstance(stop_loss, (int, float)) else stop_loss}</p>
                 <div class="strength-meter">
                     <div class="strength-meter-fill" style="width: {success_percentage}%; background: {'#22c55e' if 'BUY' in recommendation else '#ef4444' if 'SELL' in recommendation else '#6b7280'};"></div>
                 </div>
                 <p>Estimated Success Probability: {success_percentage}%</p>
             </div>
             """, unsafe_allow_html=True)


             # Add notes on interpreting options data
             st.markdown("""
             <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 20px;'>
                 <strong>Interpretation Notes:</strong>
                 <ul>
                     <li><strong>OI / Change in OI:</strong> High and increasing Call OI at a strike suggests resistance; high and increasing Put OI suggests support.</li>
                     <li><strong>PCR (OI):</strong> Generally, PCR < 1.0 is seen as bullish/oversold, while PCR > 1.0 is seen as bearish/overbought. Extreme values can signal reversals.</li>
                     <li><strong>Max Pain:</strong> The price often tends to gravitate towards the Max Pain strike near expiry.</li>
                     <li><strong>IV:</strong> Higher IV means options are more expensive, implying higher expected volatility or uncertainty. Sudden changes can be significant.</li>
                     <li><strong>Volume:</strong> High volume at a strike indicates significant activity there today.</li>
                     <li><em>Options data analysis is complex and should be used in conjunction with other indicators.</em></li>
                     <li><em>The Recommendation, Target, Stop Loss, and Success Probability are based solely on the options data provided and a simplified logic. They are not financial advice.</em></li>
                 </ul>
                 <em>Note: This analysis is based on the static data from the uploaded JSON, not real-time data.</em>
             </div>
             """, unsafe_allow_html=True)

        else:
            # This message is shown if options_results is not None but doesn't contain analysis data
            st.info("No options data processed for display after filtering. Please check the uploaded file and selected expiry.")


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
            <li>This tool uses options chain data to generate insights.</li>
            <li>Please understand that these methods are based on probability, not certainty.</li>
            <li>There is no guarantee that the predictions will be accurate.</li>
            <li>Financial risk is involved in all trading activities.</li>
            <li>I am not a SEBI-registered advisor, and this is not financial advice.</li>
            <li>Always do your own research or consult a certified financial advisor before making any investment decisions.</li>
            <li><strong>Options data is a snapshot from the uploaded JSON file and is not real-time. Analysis based on it is for informational purposes only.</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# --- Run the app ---
if __name__ == "__main__":
    main()
