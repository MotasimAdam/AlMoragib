#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المراقب الكوني (نجم الأسواق) - النسخة النهائية
Al-Muraqib Al-Kawni (Najm Al-Aswaq) - Ultimate Edition
"""

import os
import ccxt
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import time
from datetime import datetime
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
    RISK_PER_TRADE = 0.02
    MIN_CONFIDENCE = 0.75
    SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    MODEL_PATH = 'model/lstm_model.h5'
    TRAINING_EPOCHS = 100
    KELLY_FACTOR = 0.3
    LOOKBACK_PERIOD = 60

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
# إدارة المخاطر
# ---------------------------
class RiskManager:
    def __init__(self, exchange):
        self.exchange = exchange
    
    def calculate_position_size(self, symbol):
        """حساب حجم المركز باستخدام معيار كيلي"""
        balance = self.exchange.fetch_balance()['USDT']['free']
        win_rate = 0.6  # يجب تحديثها من البيانات التاريخية
        avg_win = 0.3
        avg_loss = 0.1
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / (avg_win * avg_loss)
        risk_amount = balance * min(kelly, Config.KELLY_FACTOR)
        return risk_amount / self.exchange.fetch_ticker(symbol)['last']
    
    def dynamic_stop_loss(self, df):
        """وقف الخسارة الديناميكي"""
        atr = df['atr'].iloc[-1]
        candle_size = df['high'].iloc[-1] - df['low'].iloc[-1]
        return df['close'].iloc[-1] - max(1.5 * atr, 0.03 * candle_size)
    
    def dynamic_take_profit(self, df):
        """جني الأرباح الديناميكي"""
        fib_levels = df['fib_levels'].iloc[-1]
        return df['close'].iloc[-1] + (fib_levels[1] - fib_levels[0])

# ---------------------------
# البوت الرئيسي
# ---------------------------
class TradingBot:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('API_KEY'),
            'secret': os.getenv('API_SECRET'),
            'options': {'defaultType': 'future'},
            'enableRateLimit': True
        })
        self.analyzer = MarketAnalyzer(self.exchange)
        self.pattern_detector = PatternDetector()
        self.risk_manager = RiskManager(self.exchange)
        self.model = DLModel()
    
    def analyze_symbol(self, symbol):
        """تحليل شامل لرمز التداول"""
        try:
            df = self.analyzer.get_market_data(symbol, '4h')
            df = self.analyzer.calculate_indicators(df)
            
            # الكشف عن الأنماط
            patterns = self.pattern_detector.detect_candlestick_patterns(df)
            
            # التنبؤ بالتعلم العميق
            dl_input = df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']].iloc[-Config.LOOKBACK_PERIOD:]
            dl_prediction = self.model.model.predict(np.array([dl_input.values]))
            
            # حساب الثقة
            confidence = self._calculate_confidence(df, patterns, dl_prediction)
            
            return {
                'symbol': symbol,
                'confidence': confidence,
                'action': 'buy' if confidence >= Config.MIN_CONFIDENCE else 'sell' if confidence <= -Config.MIN_CONFIDENCE else 'hold',
                'stop_loss': self.risk_manager.dynamic_stop_loss(df),
                'take_profit': self.risk_manager.dynamic_take_profit(df)
            }
        except Exception as e:
            logging.error(f"Analysis failed for {symbol}: {str(e)}")
            return None
    
    def _calculate_confidence(self, df, patterns, dl_prediction):
        """حساب درجة الثقة المركبة"""
        confidence = 0
        
        # وزن الأنماط
        pattern_weights = {
            'hammer': 0.15,
            'engulfing': 0.2,
            'rising_wedge': 0.1,
            'falling_wedge': -0.1,
            'morning_star': 0.25,
            'evening_star': -0.25
        }
        
        for pattern, detected in patterns.items():
            if detected:
                confidence += pattern_weights.get(pattern, 0)
        
        # إضافة تنبؤ النموذج
        confidence += (dl_prediction[0][2] - dl_prediction[0][0])  # صعودي - هبوطي
        
        # مراعاة موقع السعر بالنسبة للسحابة
        if df['close'].iloc[-1] > df['senkou_span_a'].iloc[-1]:
            confidence += 0.1
        
        return round(confidence, 2)
    
    def execute_trade(self, decision):
        """تنفيذ الصفقة"""
        if decision['action'] != 'hold':
            try:
                size = self.risk_manager.calculate_position_size(decision['symbol'])
                self.exchange.create_market_order(
                    symbol=decision['symbol'],
                    side=decision['action'],
                    amount=size,
                    params={
                        'stopLossPrice': decision['stop_loss'],
                        'takeProfitPrice': decision['take_profit']
                    }
                )
                logging.info(f"Executed {decision['action']} on {decision['symbol']}")
            except Exception as e:
                logging.error(f"Trade failed: {str(e)}")
    
    def run(self):
        """الدورة الرئيسية للتداول"""
        logging.info("Starting المراقب الكوني...")
        while True:
            try:
                for symbol in Config.SYMBOLS:
                    analysis = self.analyze_symbol(symbol)
                    if analysis:
                        self.execute_trade(analysis)
                time.sleep(3600)  # تشغيل كل ساعة
            except KeyboardInterrupt:
                logging.info("إيقاف البوت...")
                break

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
    bot = TradingBot()
    bot.run()
