#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المراقب الكوني (نجم الأسواق) - النسخة النهائية مع البحث التلقائي عن العملات
Al-Muraqib Al-Kawni (Najm Al-Aswaq) - Ultimate Edition with Auto Symbol Discovery
"""

import os
import ccxt
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from cryptography.fernet import Fernet
from ta.trend import IchimokuIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.utils import fibonacci_levels
from dotenv import load_dotenv

# تحميل المتغيرات البيئية
load_dotenv()

# ---------------------------
# الإعدادات العامة
# ---------------------------
class Config:
    TIMEFRAMES = ['15m', '1h', '4h']
    RISK_PER_TRADE = 0.10
    MIN_CONFIDENCE = 0.75
    MODEL_PATH = 'model/lstm_model.h5'
    TRAINING_EPOCHS = 100
    KELLY_FACTOR = 0.3
    LOOKBACK_PERIOD = 60
    MAX_DAILY_LOSS = 0.10  # 5% من الرصيد
    MIN_TRADING_VOLUME = 100000  # الحد الأدنى لحجم التداول (بالـ USDT)
    LEVERAGE = 10  # الرافعة المالية 6x

# ---------------------------
# إعداد نظام التسجيل
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)

# ---------------------------
# وحدة تحليل السوق
# ---------------------------
class MarketAnalyzer:
    def __init__(self, exchange):
        self.exchange = exchange
    
    def get_market_data(self, symbol, timeframe, limit=200):
        """جلب البيانات السعرية"""
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def calculate_indicators(self, df):
        """حساب المؤشرات الفنية"""
        # المؤشرات الأساسية
        df['rsi'] = RSIIndicator(df['close']).rsi()
        df['macd'] = MACD(df['close']).macd_diff()
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # سحابة الإيشيموكو
        ichimoku = IchimokuIndicator(df['high'], df['low'])
        df['tenkan_sen'] = ichimoku.ichimoku_conversion_line()
        df['kijun_sen'] = ichimoku.ichimoku_base_line()
        df['senkou_span_a'] = ichimoku.ichimoku_a().shift(26)
        df['senkou_span_b'] = ichimoku.ichimoku_b().shift(26)
        
        # مستويات فيبوناتشي
        swing_high = df['high'].rolling(20).max().iloc[-1]
        swing_low = df['low'].rolling(20).min().iloc[-1]
        df['fib_levels'] = [fibonacci_levels(swing_high, swing_low)] * len(df)
        
        return df

# ---------------------------
# وحدة اكتشاف الأنماط
# ---------------------------
class PatternDetector:
    @staticmethod
    def detect_candlestick_patterns(df):
        """الكشف المتقدم عن أنماط الشموع"""
        patterns = {
            'hammer': False,
            'engulfing': False,
            'rising_wedge': False,
            'falling_wedge': False,
            'morning_star': False,
            'evening_star': False
        }
        
        candles = df.iloc[-3:]
        
        # نمط المطرقة
        body = abs(candles['close'].iloc[-1] - candles['open'].iloc[-1])
        lower_shadow = candles['open'].iloc[-1] - candles['low'].iloc[-1]
        patterns['hammer'] = (lower_shadow > 2 * body) and (candles['close'].iloc[-1] > candles['open'].iloc[-1])
        
        # نمط الابتلاع
        prev_body = candles['close'].iloc[-2] - candles['open'].iloc[-2]
        current_body = candles['close'].iloc[-1] - candles['open'].iloc[-1]
        patterns['engulfing'] = (abs(current_body) > abs(prev_body)) and (current_body * prev_body < 0)
        
        # أنماط الوتد
        patterns['rising_wedge'] = (
            (candles['high'].iloc[-3] < candles['high'].iloc[-2] < candles['high'].iloc[-1]) and
            (candles['low'].iloc[-3] < candles['low'].iloc[-2] < candles['low'].iloc[-1])
        )
        
        patterns['falling_wedge'] = (
            (candles['high'].iloc[-3] > candles['high'].iloc[-2] > candles['high'].iloc[-1]) and
            (candles['low'].iloc[-3] > candles['low'].iloc[-2] > candles['low'].iloc[-1])
        )
        
        return patterns

# ---------------------------
# نموذج التعلم العميق
# ---------------------------
class DLModel:
    def __init__(self):
        self.model = self._build_lstm_model()
    
    def _build_lstm_model(self):
        """بناء نموذج LSTM"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, input_shape=(Config.LOOKBACK_PERIOD, 7), return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_model(self, historical_data):
        """تدريب النموذج على البيانات التاريخية"""
        X, y = self._preprocess_data(historical_data)
        self.model.fit(X, y, epochs=Config.TRAINING_EPOCHS, validation_split=0.2)
        self.model.save(Config.MODEL_PATH)
    
    def _preprocess_data(self, data):
        """تحضير البيانات للتدريب"""
        sequences = []
        labels = []
        features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        
        for i in range(len(data) - Config.LOOKBACK_PERIOD - 1):
            seq = data.iloc[i:i+Config.LOOKBACK_PERIOD][features]
            label = data.iloc[i+Config.LOOKBACK_PERIOD+1]['trend']
            sequences.append(seq.values)
            labels.append(label)
        
        return np.array(sequences), tf.keras.utils.to_categorical(labels, num_classes=3)

# ---------------------------
# إدارة المخاطر المحسنة للعقود الآجلة
# ---------------------------
class EnhancedRiskManager:
    def __init__(self, exchange):
        self.exchange = exchange
        self.max_daily_loss = Config.MAX_DAILY_LOSS
        self.leverage = Config.LEVERAGE

    def set_leverage(self, symbol):
        """تعيين الرافعة المالية"""
        try:
            self.exchange.set_leverage(self.leverage, symbol)
            logging.info(f"تم تعيين الرافعة المالية إلى {self.leverage}x لـ {symbol}")
        except Exception as e:
            logging.error(f"فشل في تعيين الرافعة المالية: {str(e)}")

    def check_daily_loss(self):
        """التحقق من الخسارة اليومية"""
        daily_trades = self.exchange.fetch_my_trades(since=int(time.time()) - 86400)
        total_loss = sum(trade['realized_pnl'] for trade in daily_trades if trade['realized_pnl'] < 0)
        return total_loss < self.max_daily_loss * self.exchange.fetch_balance()['USDT']['free']

    def check_liquidity(self, symbol, amount):
        """التحقق من السيولة"""
        order_book = self.exchange.fetch_order_book(symbol)
        return order_book['bids'][0][0] * amount > 1000  # 1000 USDT كحد أدنى

# ---------------------------
# البوت الرئيسي المحسن مع البحث التلقائي عن العملات
# ---------------------------
class EnhancedTradingBot:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'options': {
                'defaultType': 'future'  # استخدام العقود الآجلة
            }
        })
        self.secure_storage = SecureAPIStorage()
        self.risk_manager = EnhancedRiskManager(self.exchange)
        self.symbols = self.update_symbols()  # تحديث الأزواج عند التشغيل
    
    def get_all_symbols(self):
        """جلب جميع أزواج التداول المتاحة على المنصة"""
        markets = self.exchange.load_markets()
        return [symbol for symbol in markets if symbol.endswith('/USDT') and 'future' in markets[symbol]['info']['contractType']]
    
    def filter_liquid_symbols(self, symbols):
        """تصفية الأزواج بناءً على حجم التداول"""
        liquid_symbols = []
        for symbol in symbols:
            ticker = self.exchange.fetch_ticker(symbol)
            if ticker['quoteVolume'] > Config.MIN_TRADING_VOLUME:  # حجم تداول لا يقل عن الحد المحدد
                liquid_symbols.append(symbol)
        return liquid_symbols
    
    def update_symbols(self):
        """تحديث قائمة الأزواج بناءً على السيولة"""
        all_symbols = self.get_all_symbols()
        return self.filter_liquid_symbols(all_symbols)
    
    def analyze_symbols_concurrently(self):
        """تحليل الرموز بشكل متزامن"""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.analyze_symbol, self.symbols))
        return results
    
    def test_api_connection(self):
        """اختبار اتصال API"""
        try:
            self.exchange.fetch_ticker('BTC/USDT')
            return True
        except Exception as e:
            logging.error(f"API Connection Failed: {str(e)}")
            return False
    
    def test_model_accuracy(self):
        """اختبار دقة النموذج"""
        test_data = pd.read_csv('test_data.csv')
        X_test, y_test = DLModel()._preprocess_data(test_data)
        loss, accuracy = self.model.model.evaluate(X_test, y_test)
        return accuracy > 0.7  # دقة لا تقل عن 70%
    
    def run(self):
        """الدورة الرئيسية للتداول"""
        logging.info("Starting المراقب الكوني...")
        while True:
            try:
                self.symbols = self.update_symbols()  # تحديث الأزواج تلقائيًا
                logging.info(f"Active symbols: {self.symbols}")
                for symbol in self.symbols:
                    self.risk_manager.set_leverage(symbol)  # تعيين الرافعة المالية
                    analysis = self.analyze_symbol(symbol)
                    if analysis:
                        self.execute_trade(analysis)
                time.sleep(3600)  # تشغيل كل ساعة
            except KeyboardInterrupt:
                logging.info("إيقاف البوت...")
                break

# ---------------------------
# وحدة التخزين الآمن
# ---------------------------
class SecureAPIStorage:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)

    def encrypt(self, data):
        """تشفير البيانات"""
        return self.cipher_suite.encrypt(data.encode())

    def decrypt(self, encrypted_data):
        """فك تشفير البيانات"""
        return self.cipher_suite.decrypt(encrypted_data).decode()

# ---------------------------
# التشغيل والتهيئة
# ---------------------------
if __name__ == "__main__":
    # تهيئة النموذج
    if not os.path.exists(Config.MODEL_PATH):
        logging.info("Training initial model...")
        historical_data = pd.read_csv('historical_data.csv')
        DLModel().train_model(historical_data)
    
    # تشغيل البوت
    bot = EnhancedTradingBot()
    
    # اختبارات ما قبل التشغيل
    if bot.test_api_connection() and bot.test_model_accuracy():
        logging.info("All tests passed. Starting bot...")
        bot.run()
    else:
        logging.error("Pre-run tests failed. Exiting...")
