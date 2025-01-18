import pandas as pd
from binance.client import Client
import os
from datetime import datetime
import time
# Binance API anahtarlarınızı buraya ekleyin
api_key = 'eomyTbeauBCOKpbhXJLWNZBqFiNnOjltb49ZmXDJXeUzViB6pvBAHHrGq3RMpWWF'
api_secret = 'cfHXS8FToHkeSHc7L4QI1DtPc6kXSdLbDGml6fTqS04Fx3zTvUkQuRaz9uE6Ivqf'

client = Client(api_key, api_secret)

# Sembol ve zaman dilimi listesi
symbols = [
    "1INCHUSDT",
    "AAVEUSDT",
    "ACHUSDT",
    "ACMUSDT",
    "ADAUSDT",
    "AGIXUSDT",
    "ALGOUSDT",
    "ALICEUSDT",
    "ALPHAUSDT",
    "ANKRUSDT",
    "ANTUSDT",
    "APEUSDT",
    "API3USDT",
    "ARPAUSDT",
    "ARUSDT",
    "ATAUSDT",
    "ATOMUSDT",
    "AUDIOUSDT",
    "AVAXUSDT",
    "AXSUSDT",
    "BAKEUSDT",
    "BALUSDT",
    "BANDUSDT",
    "BATUSDT",
    "BCHUSDT",
    "BELUSDT",
    "BLZUSDT",
    "BNBUSDT",
    "BNXUSDT",
    "BTCDOMUSDT",
    "BTCUSDT",
    "BTSUSDT",
    "C98USDT",
    "CELOUSDT",
    "CELRUSDT",
    "CHRUSDT",
    "CHZUSDT",
    "COMPUSDT",
    "COTIUSDT",
    "CRVUSDT",
    "CTKUSDT",
    "CTSIUSDT",
    "CVCUSDT",
    "CVPUSDT",
    "DARUSDT",
    "DASHUSDT",
    "DEFIUSDT",
    "DENTUSDT",
    "DGBUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "DUSKUSDT",
    "DYDXUSDT",
    "EGLDUSDT",
    "ENJUSDT",
    "ENSUSDT",
    "EOSUSDT",
    "ETCUSDT",
    "ETHUSDT",
    "FILUSDT",
    "FLMUSDT",
    "FLOWUSDT",
    "FOOTBALLUSDT",
    "FTMUSDT",
    "FTTUSDT",
    "GALAUSDT",
    "GALUSDT",
    "GMTUSDT",
    "GRTUSDT",
    "GTCUSDT",
    "HBARUSDT",
    "HNTUSDT",
    "HOTUSDT",
    "ICPUSDT",
    "ICXUSDT",
    "IMXUSDT",
    "INJUSDT",
    "IOSTUSDT",
    "IOTAUSDT",
    "JASMYUSDT",
    "KAVAUSDT",
    "KLAYUSDT",
    "KNCUSDT",
    "KSMUSDT",
    "LDOUSDT",
    "LINAUSDT",
    "LINKUSDT",
    "LITUSDT",
    "LPTUSDT",
    "LRCUSDT",
    "LTCUSDT",
    "LUNA2USDT",
    "LUNAUSDT",
    "MAGICUSDT",
    "MANAUSDT",
    "MASKUSDT",
    "MATICUSDT",
    "MINAUSDT",
    "MKRUSDT",
    "MTLUSDT",
    "NEARUSDT",
    "NEOUSDT",
    "NKNUSDT",
    "OCEANUSDT",
    "OGNUSDT",
    "OMGUSDT",
    "ONEUSDT",
    "ONTUSDT",
    "OPUSDT",
    "PEOPLEUSDT",
    "QTUMUSDT",
    "RAYUSDT",
    "REEFUSDT",
    "RENUSDT",
    "RLCUSDT",
    "RNDRUSDT",
    "ROSEUSDT",
    "RSRUSDT",
    "RUNEUSDT",
    "SANDUSDT",
    "SCUSDT",
    "SFPUSDT",
    "SKLUSDT",
    "SNXUSDT",
    "SOLUSDT",
    "SPELLUSDT",
    "SRMUSDT",
    "STMXUSDT",
    "STORJUSDT",
    "STPTUSDT",
    "STXUSDT",
    "SUSHIUSDT",
    "SXPUSDT",
    "THETAUSDT",
    "TLMUSDT",
    "TOMOUSDT",
    "TRBUSDT",
    "TRXUSDT",
    "TUSDT",
    "UNFIUSDT",
    "UNIUSDT",
    "VETUSDT",
    "WAVESUSDT",
    "WOOUSDT",
    "XEMUSDT",
    "XLMUSDT",
    "XMRUSDT",
    "XRPUSDT",
    "XTZUSDT",
    "YFIUSDT",
    "YFIIUSDT",
    "ZECUSDT",
    "ZENUSDT",
    "ZILUSDT",
    "ZRXUSDT"
]

interval = Client.KLINE_INTERVAL_1HOUR

# Unix timestamp olarak tarihleri belirleyin
start_ts = int(datetime(2020, 1, 1).timestamp() * 1000)  # milliseconds
end_ts = int(datetime(2025, 1, 1).timestamp() * 1000)    # milliseconds

output_dir = "data"

os.makedirs(output_dir, exist_ok=True)

def fetch_all_klines(symbol, interval, start_ts, end_ts):
    """Tüm kline verilerini çek"""
    klines = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        temp_klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=str(current_ts),
            end_str=str(end_ts),
            limit=1000
        )
        
        if not temp_klines:
            break
            
        klines.extend(temp_klines)
        current_ts = temp_klines[-1][0] + 1  # Son kline'ın timestamp'i + 1
        time.sleep(0.1)  # Rate limit için bekleme
        
    return klines

for symbol in symbols:
    try:
        print(f"Fetching data for {symbol}")
        
        # Veriyi çek
        klines = fetch_all_klines(symbol, interval, start_ts, end_ts)
        
        if not klines:
            print(f"No data available for {symbol}")
            continue
            
        # Veri çerçevesi oluşturma
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                   'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                   'taker_buy_quote_asset_volume', 'ignore']
        df = pd.DataFrame(klines, columns=columns)
        
        # Gereksiz sütunları kaldırma
        df = df.drop(['close_time', 'quote_asset_volume', 'number_of_trades', 
                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], axis=1)
        
        # Zaman damgasını tarih formatına çevirme
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Sayısal değerleri float'a çevirme
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # CSV olarak kaydetme
        output_file = os.path.join(output_dir, f'{symbol}_data.csv')
        df.to_csv(output_file, index=False)
        print(f"Data for {symbol} saved to {output_file}")
        
        # Rate limiting için kısa bir bekleme
        time.sleep(0.5)
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        continue

print("All data fetched and saved.")