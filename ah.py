import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import glob
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import io
import logging
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import time
from collections import Counter
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class StockAnalyzer:

    def __init__(self, data_folder="Saham"):
        self.data_folder = "/home/nedquad12/AWIG/Saham"
        self.scaler = StandardScaler()
        self.model = XGBClassifier(
             objective='multi:softmax',
             num_class=3,  # BUY, SELL, HOLD
             n_estimators=100,
             random_state=42,
             use_label_encoder=False,
             eval_metric='mlogloss'
        )
        self.model_trained = False
        self.all_data = {}
        self.load_all_data()
        self.cache_folder = "cache"
        self.cache_duration = 3 * 60 * 60  # 3 hours in seconds
        self.analysis_cache = {}
        self.model_cache = {}
        self.ensure_cache_folder()

    def ensure_cache_folder(self):
        """Ensure cache folder exists"""
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

    def get_cache_key(self, stock_code, analysis_type="full"):
        """Generate cache key"""
        return f"{stock_code}_{analysis_type}_{datetime.now().strftime('%Y%m%d')}"

    def save_to_cache(self, key, data, cache_type="analysis"):
        """Save data to cache with timestamp"""
        cache_data = {'timestamp': time.time(), 'data': data}

        cache_file = os.path.join(self.cache_folder, f"{key}_{cache_type}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.error(f"Error saving cache {key}: {str(e)}")

    def load_from_cache(self, key, cache_type="analysis"):
        """Load data from cache if valid"""
        cache_file = os.path.join(self.cache_folder, f"{key}_{cache_type}.pkl")

        if not os.path.exists(cache_file):
            return None
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # Check if cache is still valid (24 hours)
            if time.time() - cache_data['timestamp'] < self.cache_duration:
                return cache_data['data']
            else:
                # Remove expired cache
                os.remove(cache_file)
                logger.info(f"Removed expired cache: {key}")
                return None

        except Exception as e:
            logger.error(f"Error loading cache {key}: {str(e)}")
            return None

    def clear_expired_cache(self):
        """Clear all expired cache files"""
        try:
            for filename in os.listdir(self.cache_folder):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(self.cache_folder, filename)
                    try:
                        with open(filepath, 'rb') as f:
                            cache_data = pickle.load(f)

                        if time.time(
                        ) - cache_data['timestamp'] >= self.cache_duration:
                            os.remove(filepath)
                            logger.info(f"Removed expired cache: {filename}")
                    except:
                        # If can't read file, remove it
                        os.remove(filepath)

        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

    def load_all_data(self):
        """Load semua data Excel dari folder Saham"""
        excel_files = glob.glob(os.path.join(self.data_folder, "*.xlsx"))
        excel_files.extend(glob.glob(os.path.join(self.data_folder, "*.xls")))

        for file_path in excel_files:
            try:
                # Extract date from filename (format ddmmyy)
                filename = os.path.basename(file_path)
                date_str = filename.split('.')[0][
                    -6:]  # Get last 6 characters before extension

                # Convert to datetime
                date = datetime.strptime(date_str, "%d%m%y")

                # Read Excel file
                df = pd.read_excel(file_path)

                # Standardize column names
                df.columns = df.columns.str.strip()

                # Store data with date
                self.all_data[date.strftime("%Y-%m-%d")] = df

                logger.info(
                    f"Loaded data from {filename} for date {date.strftime('%Y-%m-%d')}"
                )

            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")

    def get_stock_data(self, stock_code, days=260):
        """Ambil data saham untuk beberapa hari terakhir"""
        # Check cache first
        cache_key = self.get_cache_key(stock_code, f"data_{days}")
        cached_data = self.load_from_cache(cache_key, "stock_data")

        if cached_data is not None:
            return cached_data

        stock_data = []

        # Sort dates in descending order
        sorted_dates = sorted(self.all_data.keys(), reverse=True)

        for date in sorted_dates[:days]:
            if date in self.all_data:
                df = self.all_data[date]

                # Find stock in dataframe
                stock_row = df[df['Kode Saham'].str.upper() ==
                               stock_code.upper()]

                if not stock_row.empty:
                    row = stock_row.iloc[0]

                    # Extract required columns
                    data_point = {
                        'Date':
                        date,
                        'Previous':
                        self.safe_convert(row.get('Sebelumnya', 0)),
                        'Open':
                        self.safe_convert(row.get('Open Price', 0)),
                        'High':
                        self.safe_convert(row.get('Tertinggi', 0)),
                        'Low':
                        self.safe_convert(row.get('Terendah', 0)),
                        'Close':
                        self.safe_convert(row.get('Penutupan', 0)),
                        'Volume':
                        self.safe_convert(row.get('Volume', 0)),
                        'Frequency':
                        self.safe_convert(row.get('Frekuensi', 0)),
                        'Foreign_Sell':
                        self.safe_convert(row.get('Foreign Sell', 0)),
                        'Foreign_Buy':
                        self.safe_convert(row.get('Foreign Buy', 0)),
                        'First_Trade':
                        self.safe_convert(row.get('First Trade', 0))
                    }

                    stock_data.append(data_point)

        if not stock_data:
            return None

        # Convert to DataFrame and sort by date
        df = pd.DataFrame(stock_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Save to cache
        self.save_to_cache(cache_key, df, "stock_data")

        return df

    def safe_convert(self, value):
        """Safely convert value to float"""
        try:
            if pd.isna(value) or value == '' or value == '-':
                return 0.0
            return float(value)
        except:
            return 0.0

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # Moving averages
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()

        # Foreign net flow
        df['Foreign_Net'] = df['Foreign_Buy'] - df['Foreign_Sell']

        # Price change
        df['Price_Change'] = df['Close'].pct_change()

        # Volume change
        df['Volume_Change'] = df['Volume'].pct_change()

        return df

    def calculate_dynamic_levels(self, data):
        """Calculate dynamic support/resistance and volatility-based levels"""
        latest = data.iloc[-1]
        current_price = latest['Close']

        # Calculate volatility (ATR-like)
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift(1))
        low_close = abs(data['Low'] - data['Close'].shift(1))
        volatility = pd.concat([high_low, high_close, low_close],
                               axis=1).max(axis=1).rolling(14).mean().iloc[-1]

        # Support/Resistance levels from recent highs/lows
        recent_data = data.tail(50)  # Last 50 days
        resistance_levels = recent_data['High'].nlargest(3).tolist()
        support_levels = recent_data['Low'].nsmallest(3).tolist()

        # Volume-based strength
        avg_volume = data['Volume'].tail(20).mean()
        volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 1

        # Foreign flow impact
        foreign_net = latest['Foreign_Buy'] - latest['Foreign_Sell']
        avg_foreign_net = (data['Foreign_Buy'] -
                           data['Foreign_Sell']).tail(10).mean()
        foreign_strength = foreign_net / abs(
            avg_foreign_net) if avg_foreign_net != 0 else 0

        # MA trend strength
        ma7 = latest.get('MA7', current_price)
        ma20 = latest.get('MA20', current_price)
        ma_trend = (ma7 - ma20) / ma20 if ma20 != 0 else 0

        return {
            'volatility':
            volatility,
            'resistance_levels':
            resistance_levels,
            'support_levels':
            support_levels,
            'volume_ratio':
            volume_ratio,
            'foreign_strength':
            foreign_strength,
            'ma_trend':
            ma_trend,
            'trend_strength':
            abs(ma_trend) + (volume_ratio - 1) * 0.5 + foreign_strength * 0.3
        }

    def analyze_momentum(self, data):
        """Analyze price momentum and trend strength"""
        if len(data) < 10:
            return {'momentum': 0, 'strength': 'weak', 'direction': 'sideways'}

        latest = data.iloc[-1]
        current_price = latest['Close']

        # Price momentum (last 50 vs previous 50 days)
        recent_avg = data['Close'].tail(50).mean()
        previous_avg = data['Close'].iloc[-100:-50].mean()
        price_momentum = (recent_avg - previous_avg
                          ) / previous_avg if previous_avg != 0 else 0

        # Volume momentum
        recent_volume = data['Volume'].tail(50).mean()
        previous_volume = data['Volume'].iloc[-100:-50].mean()
        volume_momentum = (recent_volume - previous_volume
                           ) / previous_volume if previous_volume != 0 else 0

        # Foreign flow momentum
        recent_foreign = (data['Foreign_Buy'] -
                          data['Foreign_Sell']).tail(50).mean()
        previous_foreign = (data['Foreign_Buy'] -
                            data['Foreign_Sell']).iloc[-100:-50].mean()
        foreign_momentum = (recent_foreign - previous_foreign) / abs(
            previous_foreign) if previous_foreign != 0 else 0

        # MA slope (trend direction)
        ma7 = data.get('MA7', pd.Series([current_price] * len(data)))
        ma_slope = (ma7.iloc[-1] - ma7.iloc[-50]) / ma7.iloc[-50] if len(
            ma7) >= 50 and ma7.iloc[-50] != 0 else 0

        # Overall momentum score
        momentum_score = (price_momentum * 0.4 + volume_momentum * 0.2 +
                          foreign_momentum * 0.2 + ma_slope * 0.2)

        # Determine strength and direction
        if abs(momentum_score) > 0.05:
            strength = 'strong'
        elif abs(momentum_score) > 0.02:
            strength = 'moderate'
        else:
            strength = 'weak'

        if momentum_score > 0.01:
            direction = 'bullish'
        elif momentum_score < -0.01:
            direction = 'bearish'
        else:
            direction = 'sideways'

        return {
            'momentum': momentum_score,
            'strength': strength,
            'direction': direction,
            'price_momentum': price_momentum,
            'volume_momentum': volume_momentum,
            'foreign_momentum': foreign_momentum,
            'ma_slope': ma_slope
        }

    def train_model(self):
        """Train ML model or load from disk"""
        model_file = os.path.join(self.cache_folder, "ml_model.pkl")
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.model_trained = True
                logger.info("✅ Model loaded from disk.")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Gagal load model dari disk: {e}")

    # Train new model
        all_features = []
        all_targets = []
        stock_codes = set()
        for date, df in self.all_data.items():
            stock_codes.update(df['Kode Saham'].str.upper().tolist())

        for stock_code in list(stock_codes):
            try:
                data = self.get_stock_data(stock_code, days=100)
                if data is not None and len(data) > 20:
                    features, targets = self.prepare_features(data)
                    if features:
                        all_features.extend(features)
                        all_targets.extend(targets)
            except Exception as e:
                logger.error(f"Error training on {stock_code}: {e}")
                continue

        if not all_features:
            logger.warning("❌ Tidak ada data latih tersedia.")
            return False

        try:
            X = np.asarray(all_features, dtype=np.float64)
            y = np.asarray(all_targets, dtype=np.float64)
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[mask]
            y = y[mask]
            
            # Map label: -1 → 2 (SELL), 0 → 1 (HOLD), 1 → 0 (BUY)
            label_map = {-1: 2, 0: 1, 1: 0}
            y = np.asarray([label_map[label] for label in y], dtype=np.int32)
            
            # Tambahkan print info dataset
            print(f"📊 Total data latih: {len(X)}")
            counts = Counter(y)
            print(f"🟢 Jumlah BUY: {np.sum(y==1)}")
            print(f"🟠 Jumlah HOLD: {counts[0]}")
            print(f"🔴 Jumlah SELL: {np.sum(y==0)}")
            
            if len(X) < 10:
                return False
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.model_trained = True

            # Save to disk
            with open(model_file, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
            return True
        except Exception as e:
            logger.error(f"❌ Gagal melatih model: {e}")
            return False

    def prepare_features(self, data):
        """Prepare features for ML model"""
        features = []
        targets = []

        for i in range(20, len(data) - 5):
            try:
                # Create pattern-based features (last 50 days)
                pattern_data = data.iloc[i - 50:i]

                # Check if pattern_data is valid
                if len(pattern_data) < 50:
                    continue

        # Pattern features
                feature_vector = self.extract_pattern_features(pattern_data)

                # Check if features are valid
                if not feature_vector or any(pd.isna(feature_vector)):
                    continue

        # Multi-day target (not just next day)
                target_info = self.calculate_optimal_entry(data.iloc[i:i+5])

                if target_info['label'] == 1:  # BUY
                   features.append(feature_vector)
                   targets.append(1)
                elif target_info['label'] == 0:  # HOLD
                  features.append(feature_vector)
                  targets.append(0)
                elif target_info['label'] == -1:  # SELL
                  features.append(feature_vector)
                  targets.append(-1)


            except Exception as e:
                logger.error(f"Error processing data at index {i}: {str(e)}")
                continue

        return features, targets


    def extract_pattern_features(self, pattern_data):
        """Extract comprehensive pattern features for ML"""
        # Price patterns
        price_trend = (
            pattern_data['Close'].iloc[-1] -
            pattern_data['Close'].iloc[0]) / pattern_data['Close'].iloc[0]
        price_volatility = pattern_data['Close'].pct_change().std()
        price_momentum = pattern_data['Close'].pct_change().tail(3).mean()

        # Volume patterns
        volume_trend = (
            pattern_data['Volume'].iloc[-1] -
            pattern_data['Volume'].iloc[0]) / pattern_data['Volume'].iloc[0]
        volume_spike = pattern_data['Volume'].iloc[-1] / pattern_data[
            'Volume'].mean()
        volume_consistency = 1 - pattern_data['Volume'].pct_change().std()

        # Foreign flow patterns
        foreign_net = pattern_data['Foreign_Buy'] - pattern_data['Foreign_Sell']
        foreign_trend = (foreign_net.iloc[-1] - foreign_net.iloc[0]) / abs(
            foreign_net.iloc[0]) if foreign_net.iloc[0] != 0 else 0
        foreign_accumulation = foreign_net.sum()
        foreign_momentum = foreign_net.tail(3).mean()

        # Frequency patterns
        freq_trend = (pattern_data['Frequency'].iloc[-1] -
                      pattern_data['Frequency'].iloc[0]
                      ) / pattern_data['Frequency'].iloc[0]
        freq_activity = pattern_data['Frequency'].mean()

        # MA patterns
        ma7_slope = (
            pattern_data['MA7'].iloc[-1] - pattern_data['MA7'].iloc[0]
        ) / pattern_data['MA7'].iloc[0] if 'MA7' in pattern_data.columns else 0
        ma20_slope = (
            pattern_data['MA20'].iloc[-1] -
            pattern_data['MA20'].iloc[0]) / pattern_data['MA20'].iloc[
                0] if 'MA20' in pattern_data.columns else 0
        ma_convergence = abs(
            pattern_data['MA7'].iloc[-1] -
            pattern_data['MA20'].iloc[-1]) / pattern_data['Close'].iloc[
                -1] if 'MA7' in pattern_data.columns else 0

        # Cross-factor correlations
        price_volume_corr = pattern_data['Close'].corr(pattern_data['Volume'])
        price_foreign_corr = pattern_data['Close'].corr(foreign_net)
        volume_foreign_corr = pattern_data['Volume'].corr(foreign_net)

        return [
            price_trend, price_volatility, price_momentum, volume_trend,
            volume_spike, volume_consistency, foreign_trend,
            foreign_accumulation, foreign_momentum, freq_trend, freq_activity,
            ma7_slope, ma20_slope, ma_convergence, price_volume_corr,
            price_foreign_corr, volume_foreign_corr
        ]

        # Validate all features are numbers
        validated_features = []
        for feature in [price_trend, price_volatility, price_momentum, ...]:
            if pd.isna(feature) or not isinstance(feature, (int, float)):
                validated_features.append(0.0)
            else:
                validated_features.append(float(feature))

        return validated_features

    def calculate_optimal_entry(self, future_data):
        """Calculate optimal entry point from future data"""
        if len(future_data) < 3:
           return {'label': 0}  # HOLD

        start_price = future_data['Close'].iloc[0]
        best_profit = 0
        min_loss = 0

        for i in range(1, len(future_data)):
            entry_price = future_data['Close'].iloc[i]
            remaining_data = future_data.iloc[i:]
            if len(remaining_data) > 1:
               max_profit = (remaining_data['Close'].max() - entry_price) / entry_price
               min_loss = (remaining_data['Close'].min() - entry_price) / entry_price

        # Ubah threshold profit dan loss
               if max_profit > 0.05 and min_loss > -0.03:
                  return {'label': 1}  # BUY
               elif max_profit < 0.03 and min_loss > -0.01:
                  return {'label': 0}  # HOLD

        return {'label': -1}  # SELL


    def predict_signals(self, data, stock_code):
        """Generate buy/sell signals"""
        if data is None or len(data) == 0:
            return None

        # Check cache
        cache_key = self.get_cache_key(stock_code, "signals")
        cached_signals = self.load_from_cache(cache_key, "signals")

        if cached_signals is not None:
            return cached_signals

        if not self.model_trained:
            raise Exception(
                "⚠️ Model ML belum dilatih. Latih model sebelum menjalankan bot."
            )

        try:
            # Get latest features
            latest = data.iloc[-1]

            current_pattern = data.tail(10)
            feature_vector = self.extract_pattern_features(current_pattern)

            if feature_vector is None or len(feature_vector) != 17:
                logger.error("Feature vector invalid or wrong size")
                return self.generate_basic_signals(data)

            # Check for NaN values
            if any(pd.isna(feature_vector)):
                raise Exception("⚠️ Gagal memproses sinyal.")

            # Predict
            X = np.asarray([feature_vector], dtype=np.float64)
            X_scaled = self.scaler.transform(X)

            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
            
            # Mapping prediksi multi-class ke sinyal
            if prediction == 0:
               signal_type = "BUY"
            elif prediction == 1:
              signal_type = "HOLD"
            else:
             signal_type = "SELL"

            # Generate signals based on prediction
            current_price = latest['Close']

            # Calculate dynamic levels
            dynamic_levels = self.calculate_dynamic_levels(data)

            # Predict optimal entry
            entry_prediction = self.predict_optimal_entry(data)

            if entry_prediction.get('has_entry') and 'entry_price' in entry_prediction:
               signals = self.generate_dynamic_signals_with_entry(
               entry_prediction['entry_price'],
               entry_prediction['confidence'], data, dynamic_levels)
            else:
                logger.warning(f"⚠️ Entry skipped for {stock_code}: {entry_prediction.get('reason')}")
                signals = self.generate_dynamic_hold_signals(current_price)


        except Exception as e:
            logger.error(f"Error in ML prediction: {str(e)}")
            return self.generate_basic_signals(data)

        self.save_to_cache(cache_key, signals, "signals")

        return signals

    def predict_optimal_entry(self, data):
        """Predict optimal entry price using ML"""
        if not self.model_trained or len(data) < 50:
            return {'has_entry': False, 'reason': 'insufficient_data'}

        try:
            current_pattern = data.tail(10)
            features = self.extract_pattern_features(current_pattern)
            
            if any(pd.isna(features)):
               return {'has_entry': False, 'reason': 'invalid_features'}

        # Predict signal and confidence
            X = np.asarray([features], dtype=np.float64)
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = max(probabilities)

            # Tentukan entry price optimal
            future_window = data.tail(5)
            if prediction == 1:  # BUY
                entry_price = future_window['Low'].min()
            else:  # SELL
                entry_price = future_window['High'].max()

            return {
                'has_entry': True,
                'entry_price': entry_price,
                'confidence': confidence,
                'signal_type': 'BUY' if prediction == 1 else 'SELL'
            }

        except Exception as e:
            logger.error(f"Error predicting entry: {str(e)}")
            return {'has_entry': False, 'reason': 'prediction_error'}

    def generate_dynamic_signals_with_entry(self, entry_price, confidence,
                                            data, dynamic_levels):
        """Generate dynamic signals based on ML entry prediction"""
        signal_type = "BUY" if confidence >= 0.7 else "SELL"

        # Hitung proyeksi harga tertinggi & terendah (misal 5 hari terakhir)
        future_window = data.tail(5)
        predicted_high = future_window['High'].max()
        predicted_low = future_window['Low'].min()

        if signal_type == "BUY":
            return {
                'signal': 'BUY',
                'entry': entry_price,
                'profit_1': predicted_high * 1.02,
                'profit_2': predicted_high * 1.04,
                'profit_3': predicted_high * 1.07,
                'stop_loss_1': predicted_low * 0.98,
                'stop_loss_2': predicted_low,
                'stop_loss_3': predicted_low * 0.94,
                'target_estimate': predicted_high * 1.085
            }
        else:
            return {
                'signal': 'SELL',
                'entry': entry_price,
                'profit_1': predicted_low * 0.99,
                'profit_2': predicted_low * 0.97,
                'profit_3': predicted_low * 0.94,
                'stop_loss_1': predicted_high * 1.02,
                'stop_loss_2': predicted_high,
                'stop_loss_3': predicted_high * 1.04,
                'target_estimate': predicted_low * 0.92
            }

    def generate_dynamic_hold_signals(self, current_price):
        """Generate hold signals"""
        return {
            'signal': 'HOLD',
            'profit_1': current_price * 1.01,
            'profit_2': current_price * 1.03,
            'profit_3': current_price * 1.1,
            'stop_loss_1': current_price * 0.99,
            'stop_loss_2': current_price * 0.97,
            'stop_loss_3': current_price * 0.93,
            'target_estimate': current_price
        }

    def create_chart(self, data, stock_code, signals):
        """Create stock chart with indicators"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{stock_code.upper()} - Stock Analysis',
                     fontsize=16,
                     fontweight='bold')

        # Price chart
        ax1.plot(data['Date'],
                 data['Close'],
                 label='Close Price',
                 color='blue',
                 linewidth=2)

        # Moving averages
        if 'MA7' in data.columns:
            ax1.plot(data['Date'],
                     data['MA7'],
                     label='MA7',
                     color='orange',
                     alpha=0.7)
        if 'MA20' in data.columns:
            ax1.plot(data['Date'],
                     data['MA20'],
                     label='MA20',
                     color='red',
                     alpha=0.7)
        if 'MA200' in data.columns:
            ax1.plot(data['Date'],
                     data['MA200'],
                     label='MA200',
                     color='green',
                     alpha=0.7)

        # Signal annotation
        latest_date = data['Date'].iloc[-1]
        latest_price = data['Close'].iloc[-1]

        signal_color = 'green' if signals[
            'signal'] == 'BUY' else 'red' if signals[
                'signal'] == 'SELL' else 'orange'
        ax1.annotate(f'{signals["signal"]}',
                     xy=(latest_date, latest_price),
                     xytext=(10, 10),
                     textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor=signal_color,
                               alpha=0.7),
                     arrowprops=dict(arrowstyle='->',
                                     connectionstyle='arc3,rad=0'))

        ax1.set_title('Price Chart with Moving Averages')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Volume chart
        ax2.bar(data['Date'], data['Volume'], alpha=0.7, color='purple')
        ax2.set_title('Volume')
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)

        # Frequency chart
        ax3.bar(data['Date'], data['Frequency'], alpha=0.7, color='brown')
        ax3.set_title('Frequency')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

        # Foreign flow chart
        foreign_net = data['Foreign_Buy'] - data['Foreign_Sell']
        colors = ['green' if x >= 0 else 'red' for x in foreign_net]
        ax4.bar(data['Date'], foreign_net, alpha=0.7, color=colors)
        ax4.set_title('Foreign Net Flow (Buy - Sell)')
        ax4.set_ylabel('Foreign Net Flow')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)

        # Format x-axis
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(
                mdates.DayLocator(interval=max(1,
                                               len(data) // 10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf
    
def escape_markdown_v2(text):
    escape_chars = r"_*[]()~`>#+-=|{}.!\\"
    return ''.join(['\\' + c if c in escape_chars else c for c in text])

# Initialize analyzer
analyzer = StockAnalyzer()


 # c def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
     """Start command handler"""
     welcome_message = """
    
🏛️ Selamat datang di Stock Analysis Bot!

📊 Perintah yang tersedia:
- /c <KODE>` → Chart harga + sinyal

📈 Contoh penggunaan:
• /c BBCA - Chart Bank Central Asia

🤖 Fitur:
• Volume dan frekuensi trading
• Foreign flow analysis
• AI-powered untuk sinyal buy/sell
• Multiple profit/stop loss levels

⚠️ Disclaimer: Lakukan analisa sendiri sebelum memutuskan! Sinyal bot hanya referensi. Bot TIDAK bertanggung jawab atas kerugian apa pun.

Twitter Owner: https://x.com/saberial_link/
Telegram Owner: @Rendanggedang
    """
     await update.message.reply_text(
         escape_markdown_v2(welcome_message),
         parse_mode='MarkdownV2'
     )


async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Chart command handler"""
    if not context.args:
        await update.message.reply_text(
            "❌ Masukkan kode saham. Contoh: /c BBCA", parse_mode='Markdown')
        return

    stock_code = context.args[0].upper()

    # Send loading message
    loading_msg = await update.message.reply_text(
        f"📊 Menganalisis {stock_code}, tunggu sebentar...")

    try:
        # Get stock data
        await loading_msg.edit_text(
            f"⏳ Membuat chart untuk {stock_code}, tunggu sebentar...")
        try:
            data = analyzer.get_stock_data(stock_code)
            signals = analyzer.predict_signals(data, stock_code)
            chart_buffer = analyzer.create_chart(data, stock_code, signals)
        except Exception as e:
            await loading_msg.edit_text(f"❌ Gagal membuat chart: {e}")
            return

        if data is None or len(data) == 0:
            await loading_msg.edit_text(
                f"❌ Data untuk {stock_code} tidak ditemukan.")
            return

        # Prepare signal text
        if signals['signal'] == "BUY":
           signal_text = f"""
📊 {stock_code} - Analisis Teknikal

🎯 Sinyal: {signals['signal']}

💰 Entry: {signals['entry']:,.0f}
📈 Take Profit Levels:
TP1: {signals['profit_1']:,.0f}
TP2: {signals['profit_2']:,.0f} 
TP3: {signals['profit_3']:,.0f}

🛑 *Stop Loss:*
SL: {signals['stop_loss_1']:,.0f}

🎯 Target Estimasi: {signals['target_estimate']:,.0f}

📅 Data terakhir: {data['Date'].iloc[-1].strftime('%d/%m/%Y')}
💵 Harga terakhir: {data['Close'].iloc[-1]:,.0f}
📊 Volume: {data['Volume'].iloc[-1]:,.0f}

⚠️ Disclaimer: Lakukan analisa sendiri sebelum memutuskan! Sinyal bot hanya referensi. Bot TIDAK bertanggung jawab atas kerugian apa pun.

Twitter Owner: https://x.com/saberial_link/
Telegram Owner: @Rendanggedang
"""
        elif signals['signal'] == "HOLD":
            signal_text = f"""
📊 {stock_code} - Analisis Teknikal

🎯 Sinyal: {signals['signal']}

📈 Take Profit:
TP: {signals['profit_3']:,.0f}

🛑 *Stop Loss:*
SL: {signals['stop_loss_3']:,.0f}

🎯 Target Estimasi: Tunggu sinyal berikutnya! Untuk saat ini tetap hold, jika mencapai stop loss harus dibuang jangan berharap.

📅 Data terakhir: {data['Date'].iloc[-1].strftime('%d/%m/%Y')}
💵 Harga terakhir: {data['Close'].iloc[-1]:,.0f}
📊 Volume: {data['Volume'].iloc[-1]:,.0f}

⚠️ Disclaimer: Lakukan analisa sendiri sebelum memutuskan! Sinyal bot hanya referensi. Bot TIDAK bertanggung jawab atas kerugian apa pun.

Twitter Owner: https://x.com/saberial_link/
Telegram Owner: @Rendanggedang
"""
            
        
        else:  # SELL
           signal_text = f"""
📊 {stock_code} - Analisis Teknikal

🎯 Sinyal: {signals['signal']}

📉 Support Levels:
S1: {signals['profit_1']:,.0f}
S2: {signals['profit_2']:,.0f} 
S3: {signals['profit_3']:,.0f}

🎯 Strong Support: {signals['target_estimate']:,.0f}. Jika tertembus, siapkan parasut.

📅 Data terakhir: {data['Date'].iloc[-1].strftime('%d/%m/%Y')}
💵 Harga terakhir: {data['Close'].iloc[-1]:,.0f}
📊 Volume: {data['Volume'].iloc[-1]:,.0f}

⚠️ Disclaimer: Lakukan analisa sendiri sebelum memutuskan! Sinyal bot hanya referensi. Bot TIDAK bertanggung jawab atas kerugian apa pun.

Twitter Owner: https://x.com/saberial_link/
Telegram Owner: @Rendanggedang
"""


        # Send chart and signals
        escaped_text = escape_markdown_v2(signal_text)
        await update.message.reply_photo(
              photo=chart_buffer,
              caption=escaped_text,
              parse_mode='MarkdownV2'
         )


        await loading_msg.delete()

    except Exception as e:
        await loading_msg.edit_text(f"❌ Error: {str(e)}")
        logger.error(f"Error in chart command: {str(e)}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command handler"""
    help_text = """
📚 Bantuan Stock Analysis BotBot handlers
asyn

🔧 Perintah Utama:
- /c <KODE>` → Chart harga + sinyal

📊 *Fitur Chart (/c):
• Volume dan frekuensi trading
• Foreign flow (Buy - Sell)
• Sinyal beli/jual otomatis yang dilengkapi AI canggih

📈 *Fitur Sinyal (/s):
• Entry price
• 3 level profit target
• Target estimasi harga

🤖 *Teknologi:
• Machine Learning untuk prediksi
• Analisis teknis multi-indikator
• Data end of day dan update setiap jam 18:00 WIB

⚠️ Disclaimer: Lakukan analisa sendiri sebelum memutuskan! Sinyal bot hanya referensi. Bot TIDAK bertanggung jawab atas kerugian apa pun.

Twitter Owner: https://x.com/saberial_link/
Telegram Owner: @Rendanggedang
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def list_signals(update: Update, context: ContextTypes.DEFAULT_TYPE, signal_type: str):
    """List stocks based on signal type (BUY, SELL, HOLD)"""
    try:
        all_signals = []
        for stock_code in sorted(set(code for df in analyzer.all_data.values() for code in df['Kode Saham'].unique())):
            data = analyzer.get_stock_data(stock_code)
            if data is None or len(data) == 0:
                continue
            signals = analyzer.predict_signals(data, stock_code)
            if signals['signal'] == signal_type:
                all_signals.append(stock_code)

        if not all_signals:
            await update.message.reply_text(f"⚠️ Tidak ada saham dengan sinyal {signal_type} saat ini.")
            return

        # Save list to user data for pagination
        context.user_data['signal_list'] = all_signals
        context.user_data['signal_type'] = signal_type
        context.user_data['page'] = 0

        await send_signal_page(update, context)
    except Exception as e:
        logger.error(f"Error listing signals: {e}")
        await update.message.reply_text(f"❌ Error: {e}")

async def send_signal_page(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send one page of stock signals"""
    signal_list = context.user_data.get('signal_list', [])
    signal_type = context.user_data.get('signal_type', '')
    page = context.user_data.get('page', 0)
    page_size = 5
    total_pages = (len(signal_list) + page_size - 1) // page_size

    start = page * page_size
    end = start + page_size
    stocks_on_page = signal_list[start:end]

    text = f"📊 *Saham dengan sinyal {signal_type}*\n\n"
    for idx, stock in enumerate(stocks_on_page, start=1 + start):
        text += f"{idx}. `{stock}`\n"

    keyboard = []
    if page > 0:
        keyboard.append(InlineKeyboardButton("⬅️ Previous", callback_data="prev_page"))
    if page < total_pages - 1:
        keyboard.append(InlineKeyboardButton("➡️ Next", callback_data="next_page"))

    reply_markup = InlineKeyboardMarkup([keyboard]) if keyboard else None

    if update.callback_query:
        await update.callback_query.edit_message_text(text, parse_mode="Markdown", reply_markup=reply_markup)
    else:
        await update.message.reply_text(text, parse_mode="Markdown", reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle pagination buttons"""
    query = update.callback_query
    if query.data == "next_page":
        context.user_data['page'] += 1
        await send_signal_page(update, context)
    elif query.data == "prev_page":
        context.user_data['page'] -= 1
        await send_signal_page(update, context)
    await query.answer()


def main():
    """Main function to run the bot"""
    # Train the model first
    print("🤖 Training ML model (multiple iterations)...")
    success = False
    for i in range(15):  # Latih model 15x
        print(f"📚 Training iteration {i+1}")
        success = analyzer.train_model()
        if success:
            break
    if success:
        print("✅ Model trained successfully!")
    else:
        raise Exception("❌ Model gagal dilatih. Cek data dan coba lagi.")

    # Initialize bot
    BOT_TOKEN = "7833221115:AAFwIfpm78I7AhCuwyz91XIhjsGa-eKo0ws"  # Replace with your actual bot token

    application = Application.builder().token(BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("c", chart_command))
    application.add_handler(MessageHandler(filters.ALL, button_callback))

    # Error handler
    async def error_handler(update: object,
                            context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"Exception while handling an update: {context.error}")

    application.add_error_handler(error_handler)

    # Run the bot
    print("🚀 Bot started! Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
    
    logger.info