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
    MAX_DAILY_LOSS = 0.10  # 5% من الرصيد
    MIN_TRADING_VOLUME = 100000  # الحد الأدنى لحجم التداول (بالـ USDT)
    LEVERAGE = 10  # الرافعة المالية 10x

# إضافة معاملات للتحقق من صحة البيانات
class DataValidator:
    @staticmethod
    def validate_data(df):
        if df.isnull().any().any():
            logging.error("Data contains null values.")
            return False
        if df.duplicated(subset='timestamp').any():
            logging.error("Data contains duplicate timestamps.")
            return False
        return True

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

# إضافة تسجيل للأخطاء الغير متوقعة
def setup_exception_handler():
    import sys
    sys.excepthook = lambda *ex: logging.exception("Unhandled exception", exc_info=ex)

setup_exception_handler()

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
        if not DataValidator.validate_data(df):  # تحقق من صحة البيانات
            return None
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

# إضافة تحقق من الاتجاه
class TrendDetector:
    @staticmethod
    def detect_trend(df):
        """تحديد الاتجاه (صعودي أو هبوطي أو جانبي)"""
        if df['close'].iloc[-1] > df['close'].iloc[-Config.LOOKBACK_PERIOD]:
            return 1  # صعودي
        elif df['close'].iloc[-1] < df['close'].iloc[-Config.LOOKBACK_PERIOD]:
            return 0  # هبوطي
        else:
            return 2  # جانبي

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

# إضافة نموذج للتحليل الفني
class TechnicalAnalyzer:
    @staticmethod
    def analyze_technical(df):
        """تحليل فني متكامل"""
        market = MarketAnalyzer(None)  # تمرير None لأننا لا نحتاج إلى exchange هنا
        df = market.calculate_indicators(df)
        patterns = PatternDetector.detect_candlestick_patterns(df)
        trend = TrendDetector.detect_trend(df)
        return {
            'indicators': df.iloc[-1].to_dict(),
            'patterns': patterns,
            'trend': trend
        }

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
        historical_data['trend'] = historical_data.apply(lambda row: TrendDetector.detect_trend(row), axis=1)
        X, y = self._preprocess_data(historical_data)
        self.model.fit(X, y, epochs=Config.TRAINING_EPOCHS, validation_split=0.2, verbose=1)
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
        total_loss = sum(trade['realizedPnl'] for trade in daily_trades if trade['realizedPnl'] < 0)
        balance = self.exchange.fetch_balance()
        if 'USDT' not in balance or 'free' not in balance['USDT']:
            logging.error("Could not fetch USDT balance.")
            return False
        return total_loss < self.max_daily_loss * balance['USDT']['free']

    def check_liquidity(self, symbol, amount):
        """التحقق من السيولة"""
        order_book = self.exchange.fetch_order_book(symbol)
        if not order_book['bids']:
            logging.error(f"No bids in the order book for {symbol}")
            return False
        return order_book['bids'][0][0] * amount > 1000  # 1000 USDT كحد أدنى

# إضافة تنفيذ تداولات حسب النتائج
class TradeExecutor:
    def __init__(self, exchange):
        self.exchange = exchange

    def execute_trade(self, analysis, symbol):
        if analysis['confidence'] < Config.MIN_CONFIDENCE:
            logging.info(f"Confidence too low for {symbol}. Skipping trade.")
            return
        
        direction = 'buy' if analysis['prediction'] == 'long' else 'sell'
        amount = self.calculate_amount(analysis['confidence'])
        
        try:
            order = self.exchange.create_market_order(symbol, direction, amount)
            logging.info(f"Order executed: {direction} {amount} {symbol}")
        except Exception as e:
            logging.error(f"Failed to execute trade for {symbol}: {str(e)}")

    def calculate_amount(self, confidence):
        # هنا يمكنك تطبيق استراتيجية حجم التداول مثل Kelly Criterion
        balance = self.exchange.fetch_balance()['USDT']['free']
        return balance * Config.RISK_PER_TRADE * confidence * Config.KELLY_FACTOR

# ---------------------------
# البوت الرئيسي المحسن مع البحث التلقائي عن العملات
# ---------------------------
class EnhancedTradingBot:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'options': {
                'defaultType': 'future'  # استخدام العقود الآجلة
            }
        })
        self.secure_storage = SecureAPIStorage()
        self.risk_manager = EnhancedRiskManager(self.exchange)
        self.trade_executor = TradeExecutor(self.exchange)
        self.symbols = self.update_symbols()  # تحديث الأزواج عند التشغيل
        self.model = DLModel()  # إضافة نموذج التعلم العميق
    
    def get_all_symbols(self):
        """جلب جميع أزواج التداول المتاحة على المنصة"""
        markets = self.exchange.load_markets()
        return [symbol for symbol in markets if symbol.endswith('/USDT') and 'future' in markets[symbol]['info']['contractType']]
    
    def filter_liquid_symbols(self, symbols):
        """تصفية الأزواج بناءً على حجم التداول"""
        liquid_symbols = []
        for symbol in symbols:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                if ticker['quoteVolume'] > Config.MIN_TRADING_VOLUME:  # حجم تداول لا يقل عن الحد المحدد
                    liquid_symbols.append(symbol)
            except Exception as e:
                logging.error(f"Error fetching ticker for {symbol}: {str(e)}")
        return liquid_symbols
    
    def update_symbols(self):
        """تحديث قائمة الأزواج بناءً على السيولة"""
        all_symbols = self.get_all_symbols()
        return self.filter_liquid_symbols(all_symbols)
    
    def analyze_symbol(self, symbol):
        """تحليل رمز واحد"""
        analyzer = MarketAnalyzer(self.exchange)
        for timeframe in Config.TIMEFRAMES:
            df = analyzer.get_market_data(symbol, timeframe)
            if df is None:
                logging.warning(f"Could not fetch data for {symbol} on {timeframe}. Skipping analysis.")
                return None
            
            technical_analysis = TechnicalAnalyzer.analyze_technical(df)
            
            # التنبؤ باستخدام النموذج
            X, _ = self.model._preprocess_data(df)
            prediction = self.model.model.predict(X[-1:])  # التنبؤ باستخدام آخر تسلسل
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)  # الثقة في التنبؤ

            # تحويل الفئة إلى اتجاه
            direction_map = {0: 'short', 1: 'long', 2: 'neutral'}
            prediction_direction = direction_map[predicted_class]
            
            # التحقق من السيولة قبل تنفيذ التداول
            amount = self.trade_executor.calculate_amount(confidence)  # تقدير الكمية المقصودة للتداول
            if not self.risk_manager.check_liquidity(symbol, amount):
                logging.warning(f"Insufficient liquidity for {symbol}. Skipping trade.")
                return None

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'prediction': prediction_direction,
                'confidence': confidence,
                'technical_analysis': technical_analysis
            }
    
    def analyze_symbols_concurrently(self):
        """تحليل الرموز بشكل متزامن"""
        with ThreadPoolExecutor(max_workers=10) as executor:  # Limit number of concurrent threads
            return list(executor.map(self.analyze_symbol, self.symbols))
    
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
        if not os.path.exists('test_data.csv'):
            logging.error("Test data file not found.")
            return False
        
        test_data = pd.read_csv('test_data.csv')
        if not DataValidator.validate_data(test_data):
            logging.error("Test data is invalid.")
            return False
        
        X_test, y_test = self.model._preprocess_data(test_data)
        loss, accuracy = self.model.model.evaluate(X_test, y_test)
        return accuracy > 0.7  # دقة لا تقل عن 70%
    
    def run(self):
        """الدورة الرئيسية للتداول"""
        logging.info("Starting المراقب الكوني...")
        while True:
            try:
                if not self.risk_manager.check_daily_loss():
                    logging.info("Daily loss limit reached. Shutting down...")
                    break
                
                self.symbols = self.update_symbols()  # تحديث الأزواج تلقائيًا
                logging.info(f"Active symbols: {self.symbols}")
                
                analyses = self.analyze_symbols_concurrently()
                for analysis in analyses:
                    if analysis:
                        # التحقق من السيولة مرة أخرى قبل تنفيذ التداول
                        amount = self.trade_executor.calculate_amount(analysis['confidence'])
                        if self.risk_manager.check_liquidity(analysis['symbol'], amount):
                            self.trade_executor.execute_trade(analysis, analysis['symbol'])
                        else:
                            logging.warning(f"Insufficient liquidity for {analysis['symbol']} at execution time. Skipping trade.")
                
                time.sleep(3600)  # تشغيل كل ساعة
            except KeyboardInterrupt:
                logging.info("إيقاف البوت...")
                break
            except Exception as e:
                logging.exception(f"An error occurred in the main loop: {e}")

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
        if not os.path.exists('historical_data.csv'):
            logging.error("Historical data file not found. Model cannot be trained.")
            exit(1)
        historical_data = pd.read_csv('historical_data.csv')
        if not DataValidator.validate_data(historical_data):
            logging.error("Historical data is invalid. Model cannot be trained.")
            exit(1)
        DLModel().train_model(historical_data)
    
    # تشغيل البوت
    bot = EnhancedTradingBot()
    
    # اختبارات ما قبل التشغيل
    if bot.test_api_connection() and bot.test_model_accuracy():
        logging.info("All tests passed. Starting bot...")
        bot.run()
    else:
        logging.error("Pre-run tests failed. Exiting...")
