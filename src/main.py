# Import necessary libraries
from iqoptionapi.stable_api import IQ_Option
from dotenv import load_dotenv
import pyautogui
import time
import pandas as pd
import numpy as np
import joblib
import requests
import speedtest
import logging
import os

#------------------------------------------------------------------------------------------#

# Initialize IQ option

email = os.getenv("EMAIL")
password = os.getenv("PASSWORD")

pair = "EURUSD"
timeframe = 1
amount_percentage = 5 # amount percentage 1%

target_profit = 1.5 # 1.5% of the balance
acccount_type = "PRACTICE" # "PRACTICE" or "REAL"

# Position action
amount_pos = (1880,174)
call_pos = (1845,485)
put_pos = (1845,615)
new_pos = (1840, 480)
close_pos = (295,52)

#------------------------------------------------------------------------------------------#

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Set the log level to capture all logs (DEBUG and above)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.StreamHandler(),  # This sends logs to the console
        logging.FileHandler(f'Logs/_1minute/{pair}.log', mode='a')  # This writes logs to a file
    ]
)

def log_and_print(message):
    # Log the message with INFO level
    logging.info(message)
    # Print the message to console as well
    print(message)

#------------------------------------------------------------------------------------------#

# Connect IQ option

try:
    API = IQ_Option(email, password)
    API.connect()

    if API.check_connect():
        log_and_print(f"Connected to {email}")
        log_and_print(f"Current balance: {API.get_balance()}")
    else:
        log_and_print("Error connecting. Retrying...")
        API.connect()
        if API.check_connect():
            log_and_print(f"Connected to {email} on retry")
            log_and_print(f"Current balance: {API.get_balance()}")
        else:
            log_and_print("Error to connect after retry")
            exit()
except Exception as e:
    log_and_print(f"An error occurred: {e}")

#------------------------------------------------------------------------------------------#

# Initial balance and amount

balance = API.get_balance()
amount = balance*(amount_percentage/100)
API.change_balance(acccount_type)
log_and_print(f"Acount type: {acccount_type}")

#------------------------------------------------------------------------------------------#

# Auto click functional

def move_and_click(position):
    pyautogui.moveTo(position)
    time.sleep(0.5)
    pyautogui.click()

def fill_or_edit_amount(new_amount):
    move_and_click(amount_pos)
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.typewrite(str(new_amount))
    # pyautogui.press('enter')

def get_latest_data(pair, timeframe,count=100,API=API):
    try:
        candles = API.get_candles(pair, timeframe*30, count, time.time())
        if candles:
            candle_df = pd.DataFrame(candles)
            candle_df = candle_df.drop(columns=['id','from','at','to','volume'])
            candle_df.rename(columns={'open': 'Open', 'close':'Close','min':'Low','max':'High'}, inplace=True)
            return candle_df[['High','Low','Close']]
        else:
            log_and_print("No data received.")
            return None
    except Exception as e:
        log_and_print(f"An error occurred while fetching the latest data: {e}")
        return None
    
#------------------------------------------------------------------------------------------#

# Logic functional

def calculate_ma(prices, period, method='sma'):
    if method == 'sma':
        return prices.rolling(window=period).mean()
    elif method == 'ema':
        return prices.ewm(span=period, adjust=False).mean()
    elif method == 'wma':
        weights = np.arange(1, period + 1)
        return prices.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def generate_signals(df, ma_fast_period=1, ma_slow_period=34, signal_period=5):
    data = df.copy()
    data['HL_Avg'] = (data['High'] + data['Low']) / 2

    # Calculate MAs
    data['Fast_MA'] = calculate_ma(data['HL_Avg'], ma_fast_period)
    data['Slow_MA'] = calculate_ma(data['HL_Avg'], ma_slow_period)
    data['Buffer1'] = data['Fast_MA'] - data['Slow_MA']
    data['Signal_Line'] = calculate_ma(data['Buffer1'], signal_period, method='wma')
    
    signals = []
    for i in range(len(data)):
        if i == 0 or pd.isna(data['Signal_Line'].iloc[i]) or pd.isna(data['Buffer1'].iloc[i]):
            signals.append("Hold")
            continue
        
        buffer1 = data['Buffer1'].iloc[i]
        buffer1_prev = data['Buffer1'].iloc[i - 1]
        signal_line = data['Signal_Line'].iloc[i]
        signal_line_prev = data['Signal_Line'].iloc[i - 1]
        
        if buffer1 > signal_line and buffer1_prev < signal_line_prev:
            signals.append("Call")
        elif buffer1 < signal_line and buffer1_prev > signal_line_prev:
            signals.append("Put")
        else:
            signals.append("Hold")
    
    data['Signal'] = signals
    return data['Signal'].iloc[-1]

def calculate_sma_signals(data, price_column='Close', fast_period=3, slow_period=8):
    if price_column not in data.columns:
        raise ValueError(f"Column '{price_column}' not found in DataFrame.")
    
    # Calculate EMAs instead of using pandas_ta
    data['SMA_Fast'] = calculate_ma(data[price_column], fast_period, method='ema')
    data['SMA_Slow'] = calculate_ma(data[price_column], slow_period, method='sma')
    data['diff'] = data['SMA_Fast'] - data['SMA_Slow']

    sigger = 'Hold'
    if (data['diff'].iloc[-5] < 0 and 
        data['diff'].iloc[-4] < 0 and 
        data['diff'].iloc[-3] <= 0 and 
        data['diff'].iloc[-2] > 0 and 
        data['diff'].iloc[-1] > 0):
        sigger = 'Call'
    elif (data['diff'].iloc[-5] > 0 and 
        data['diff'].iloc[-4] > 0 and 
        data['diff'].iloc[-3] >= 0 and 
        data['diff'].iloc[-2] < 0 and 
        data['diff'].iloc[-1] < 0):
        sigger = 'Put'

    # log_and_print('Diff:', data['diff'].iloc[-2], data['diff'].iloc[-1])
    return sigger

def signal_logical(pair,timeframe=1):
    df = get_latest_data(pair, timeframe,count=65,API=API)
    final_sig = 'Hold'
    sma_sig = 'Hold'
    gen_sig = 'Hold'
    gen_buffer = 'Hold'

    while True:
        gen_buffer = generate_signals(df)
        if gen_buffer!='Hold' or gen_sig!=gen_buffer:
            gen_sig = gen_buffer

        sma_sig = calculate_sma_signals(df)
        if sma_sig!='Hold' and sma_sig==gen_sig:
            final_sig = sma_sig
            break

        time.sleep(0.5)
        df = get_latest_data(pair, timeframe,count=65,API=API)

    return final_sig
    
#------------------------------------------------------------------------------------------#

# ML functional

model_path = f'../Models/unstable/iq_{pair}_{timeframe}m_model.pkl'
loaded_model = joblib.load(model_path)

def prepare_features(data):
    # Create lag features
    for lag in range(1, 6):
        # data[f'Open_lag_{lag}'] = data['Open'].shift(lag)
        data[f'High_lag_{lag}'] = data['High'].shift(lag)
        data[f'Low_lag_{lag}'] = data['Low'].shift(lag)
        data[f'Close_lag_{lag}'] = data['Close'].shift(lag)

    # Create technical indicators
    data['SMA'] = data['Close'].rolling(window=7).mean()
    data['EMA'] = data['Close'].ewm(span=5, adjust=False).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=9).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=9).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = data['Close'].ewm(span=9, adjust=False).mean()
    exp2 = data['Close'].ewm(span=13, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']

    # Drop rows with NaN values created by shifting
    data.dropna(inplace=True)
    return data

def get_signal(pair, timeframe, logic, model=loaded_model):
    df = get_latest_data(pair, timeframe*2)
    data = prepare_features(df)
    data = data.dropna()
    # prob = model.predict_proba(data)
    act = model.predict(data)
    # latest_prob = prob[-1][1]
    # log_and_print(f"Latest probability: {latest_prob}")
    model_act = act[-1]
    log_and_print("Model: ",model_act)

    action = 'hold'
    if model_act==0 and logic=='Put':
        action = "put"
    elif model_act==1 and logic=='Call':
        action = "call"
    return action, logic  


BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram_message(message,BOT_TOKEN=BOT_TOKEN,CHAT_ID=CHAT_ID):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    response = requests.post(url, data={"chat_id": CHAT_ID, "text": message})

def wait_for_next_candle():
    current_time = time.time()
    seconds = float(current_time)
    minutes = seconds//60
    next_minute = (minutes+1)*60
    time.sleep(next_minute-current_time)
    time.sleep(0.2)

def trade_bot(pair, timeframe, amount, target_profit, API=API):
    current_balance = API.get_balance()
    target_balance = current_balance*(target_profit+1)
    log_and_print(f"Current balance: {current_balance:.2f}")
    log_and_print(f"Target balance: {target_balance:.2f}")
    log_and_print(f"Amount: {amount}")
    log_and_print(f"Timeframe: {timeframe}m")
    #--------------------------------------------------#
    log_and_print(f"Starting {pair} trading bot...")
    amounts = str(amount)
    send_telegram_message(f"Starting {pair} trading bot...")
    send_telegram_message(f"Amount: {amount}")

    time.sleep(20)
    fill_or_edit_amount(amounts)
    wait_for_next_candle()
    #--------------------------------------------------#
    while current_balance < target_balance:
        buffer_balance = current_balance
        logical = signal_logical(pair,timeframe)
        wait_for_next_candle()
        signal, prob = get_signal(pair, timeframe, logical)
        log_and_print(f"Received Signal: {signal}, Logical: {prob}")
        if signal != 'hold':
            if signal == 'call':
                move_and_click(call_pos)
            elif signal == 'put':
                move_and_click(put_pos)
            else:
                log_and_print("Invalid signal")

            send_telegram_message(f"Signal: {signal}")
            time.sleep(35)
            move_and_click(new_pos)
            time.sleep(5)
            move_and_click(close_pos)
            wait_for_next_candle()
            
            current_balance = API.get_balance()
            if current_balance < buffer_balance:
                send_telegram_message(f"Trade lost, Current balance: {current_balance:.2f}")
                log_and_print(f"Trade lost, Current balance: {current_balance:.2f}")
            elif current_balance > buffer_balance:
                send_telegram_message(f"Trade won, Current balance: {current_balance:.2f}")
                log_and_print(f"Trade won, Current balance: {current_balance:.2f}")
            else:
                log_and_print(f"Trade draw.")
                send_telegram_message(f"Trade draw.")

        else:
            log_and_print("No signal received. Holding...")
            wait_for_next_candle()

#------------------------------------------------------------------------------------------#

# Main function

def check_internet_speed():
    st = speedtest.Speedtest()
    st.get_best_server()
    download_speed = st.download() / 1_000_000   # convert to Mbps
    upload_speed = st.upload() / 1_000_000       # convert to Mbps
    ping = st.results.ping

    log_and_print(f"Download Speed: {download_speed:.2f} Mbps")
    log_and_print(f"Upload Speed: {upload_speed:.2f} Mbps")
    log_and_print(f"Ping: {ping:.2f} ms")

if __name__ == "__main__":
    # check_internet_speed()
    trade_bot(pair, timeframe, amount, target_profit)
    send_telegram_message(f"Trading {pair} is Successful!.")
    log_and_print("Trading bot stopped.")
    