import os
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from datetime import date


class DataDownloader:
    def __init__(self, username, password, symbol_file, output_dir, max_workers=5, logger=None):
        """
        Initialize the DataDownloader.

        Args:
            username (str): Username for tvDatafeed.
            password (str): Password for tvDatafeed.
            symbol_file (str): Path to the CSV file containing symbols.
            output_dir (str): Directory to save downloaded data.
            max_workers (int): Maximum number of threads for concurrent downloads.
            logger (logging.Logger): Logger for logging messages.
        """
        self.username = username
        self.password = password
        self.symbol_file = symbol_file
        self.output_dir = output_dir
        self.tv = TvDatafeed(username, password)
        self.symbols = None
        self.data = []
        self.failed_symbols = []
        self.max_workers = max_workers
        self.logger = logger or logging.getLogger("DataDownloader")

    def load_symbols(self):
        """Load symbols from the given CSV file."""
        try:
            self.symbols = pd.read_csv(self.symbol_file)
            if "Symbol" not in self.symbols.columns:
                raise ValueError("Missing 'Symbol' column in the symbol file.")
            self.logger.info(f"Loaded {len(self.symbols)} symbols for download.")
        except Exception as e:
            self.logger.error(f"Error loading symbols: {e}")
            raise

    def _download_symbol_data(self, symbol, n_bars, max_retries=5):
        """
        Download data for a single symbol with retry and exponential backoff.
        """
        retries = 0
        while retries < max_retries:
            try:
                ts = self.tv.get_hist(symbol=symbol, interval=Interval.in_daily, n_bars=n_bars)
                if ts is not None:
                    self.logger.info(f"Downloaded data for symbol: {symbol}")
                    return ts, None
                else:
                    self.logger.warning(f"Received None for symbol {symbol}. Retrying...")
            except Exception as e:
                if "429 Too Many Requests" in str(e):
                    wait_time = 2 ** retries  # Exponential backoff
                    self.logger.warning(f"Rate limit hit for {symbol}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Error downloading {symbol}: {e}")
                    return None, e
            retries += 1

        self.logger.error(f"Max retries reached for {symbol}.")
        return None, "Max retries reached"

    def download_data(self, n_bars=800):
        """Download data concurrently for all symbols with rate limit handling."""
        self.logger.info(f"Starting data download for {len(self.symbols)} symbols...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._download_symbol_data, symbol, n_bars): symbol
                for symbol in self.symbols['Symbol']
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    ts, error = future.result()
                    if ts is not None:
                        self.data.append(ts)
                    if error:
                        self.failed_symbols.append(symbol)
                except Exception as e:
                    self.failed_symbols.append(symbol)
                    self.logger.error(f"Unexpected error for {symbol}: {e}")

        self.logger.info(f"Download completed. {len(self.data)} symbols downloaded successfully.")
        self.logger.warning(f"{len(self.failed_symbols)} symbols failed to download.")

    def retry_failed_downloads(self, n_bars=210):
        """Retry downloads for failed symbols."""
        if not self.failed_symbols:
            self.logger.info("No failed symbols to retry.")
            return

        self.logger.info(f"Retrying download for {len(self.failed_symbols)} failed symbols...")
        retries = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._download_symbol_data, symbol, n_bars): symbol
                for symbol in self.failed_symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    ts, error = future.result()
                    if ts is not None:
                        self.data.append(ts)
                    if error:
                        retries.append(symbol)
                except Exception as e:
                    retries.append(symbol)
                    self.logger.error(f"Unexpected error for {symbol}: {e}")

        self.failed_symbols = retries
        self.logger.warning(f"Retry completed. {len(self.failed_symbols)} symbols still failed.")

    def save_failed_symbols(self):
        """Save failed symbols to a file."""
        if self.failed_symbols:
            failed_file = os.path.join(self.output_dir, "failed_symbols.txt")
            with open(failed_file, "w") as f:
                for symbol in self.failed_symbols:
                    f.write(f"{symbol}\n")
            self.logger.warning(f"Failed symbols saved to {failed_file}")

    def save_data(self):
        """
        Save the downloaded data as both a pickle file and a CSV file.
        """
        try:
            if not self.data:
                self.logger.warning("No data to save. Exiting save_data.")
                return

            today = date.today().strftime("%Y-%m-%d")
            os.makedirs(self.output_dir, exist_ok=True)
            pickle_path = os.path.join(self.output_dir, f"TV_Daily_{today}.pkl")
            csv_path = os.path.join(self.output_dir, f"TV_Daily_{today}.csv")

            combined_data = pd.concat(self.data, ignore_index=True)
            if combined_data.empty:
                self.logger.warning("Combined data is empty. Nothing to save.")
                return

            combined_data = combined_data.rename(
                columns={
                    'datetime': 'DateTime', 'symbol': 'Symbol',
                    'open': 'Open', 'high': 'High',
                    'low': 'Low', 'close': 'Close',
                    'volume': 'Volume'
                }
            )

            combined_data.to_pickle(pickle_path)
            self.logger.info(f"Data saved to pickle file: {pickle_path}")

            combined_data.to_csv(csv_path, index=False)
            self.logger.info(f"Data saved to CSV file: {csv_path}")
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")

    def download_and_save(self):
        """Full workflow: load symbols, download data, retry failed, and save."""
        self.load_symbols()
        self.download_data()
        self.retry_failed_downloads()
        self.save_failed_symbols()
        self.save_data()


import os
import pandas as pd
import logging

class MarketAnalysis:
    def __init__(self, input_dir, output_dir):
        """
        Initialize the MarketAnalysis class.

        Args:
            input_dir (str): Directory containing the data file.
            output_dir (str): Directory to save analysis results.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.data = None
        self.results = None
        self.stock_lists = None
        self.logger = logging.getLogger("MarketAnalysis")

    def load_data(self, data_file):
        """Load the data file (CSV or pickle)."""
        try:
            file_path = os.path.join(self.input_dir, data_file)
            if data_file.endswith(".pkl"):
                self.data = pd.read_pickle(file_path)
            elif data_file.endswith(".csv"):
                self.data = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format. Use .pkl or .csv")

            self.logger.info(f"Data loaded successfully from {data_file}")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def process_data(self):
        """
        Process the data to calculate DayChangePct and identify top gainers and losers.
        """
        try:
            if self.data is None or self.data.empty:
                self.logger.error("No data loaded. Cannot process.")
                return

            if 'Open' not in self.data.columns or 'Close' not in self.data.columns:
                self.logger.error("Missing 'Open' or 'Close' columns in the data.")
                return

            # Calculate DayChangePct
            self.data['DayChangePct'] = ((self.data['Close'] / self.data['Open']) - 1) * 100

            # Filter latest date for analysis
            self.data['DateTime'] = pd.to_datetime(self.data['DateTime'])
            self.data['Date'] = self.data['DateTime'].dt.date
            latest_date = self.data['Date'].max()
            latest_data = self.data[self.data['Date'] == latest_date]

            if latest_data.empty:
                self.logger.warning(f"No data for the latest date: {latest_date}")
                return

            # Identify top gainers and losers
            self.stock_lists = {
                "5PCT_UP": latest_data[(latest_data['DayChangePct'] >= 5) & (latest_data['DayChangePct'] < 10)][
                    ['Symbol', 'Close', 'DayChangePct']
                ].round(1).to_dict('records'),
                "10PCT_UP": latest_data[latest_data['DayChangePct'] >= 10][
                    ['Symbol', 'Close', 'DayChangePct']
                ].round(1).to_dict('records'),
                "5PCT_DOWN": latest_data[(latest_data['DayChangePct'] <= -5) & (latest_data['DayChangePct'] > -10)][
                    ['Symbol', 'Close', 'DayChangePct']
                ].round(1).to_dict('records'),
                "10PCT_DOWN": latest_data[latest_data['DayChangePct'] <= -10][
                    ['Symbol', 'Close', 'DayChangePct']
                ].round(1).to_dict('records'),
            }

            self.logger.info(f"Processed data for top gainers and losers on {latest_date}")
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            raise

    def save_analysis(self):
        """
        Save the analysis results and stock lists.
        """
        try:
            if not self.stock_lists:
                self.logger.warning("No stock lists to save.")
                return

            today = pd.Timestamp.today().strftime("%b_%d_%Y")
            stock_list_dir = os.path.join(self.output_dir, f"StockLists_{today}")
            os.makedirs(stock_list_dir, exist_ok=True)

            for category, stocks in self.stock_lists.items():
                file_path = os.path.join(stock_list_dir, f"{category}.csv")
                pd.DataFrame(stocks).to_csv(file_path, index=False)
                self.logger.info(f"Saved {category} to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving stock lists: {e}")
            raise

import os
import matplotlib.pyplot as plt
import pandas as pd
import logging

class MarketPlots:
    def __init__(self, stock_lists, output_dir):
        """
        Initialize the MarketPlots class.

        Args:
            stock_lists (dict): Dictionary containing categorized stock lists.
            output_dir (str): Directory to save the bar charts.
        """
        self.stock_lists = stock_lists
        self.output_dir = output_dir
        self.logger = logging.getLogger("MarketPlots")
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_bar_chart(self, category, stocks):
        """
        Create a horizontal bar chart for a specific category.

        Args:
            category (str): The category name (e.g., "5PCT_UP").
            stocks (list of dict): List of stocks with 'Symbol', 'Close', and 'DayChangePct'.
        """
        try:
            if not stocks:
                self.logger.warning(f"No stocks available for category '{category}'. Skipping plot.")
                return

            df = pd.DataFrame(stocks).sort_values(by="DayChangePct", ascending="DOWN" in category)

            # Mobile-friendly aspect ratio configuration
            width = 6  # Smaller width for better viewing on mobile
            height = max(len(df), 5) * 0.4  # Adjust height dynamically for mobile

            plt.figure(figsize=(width, height))
            plt.barh(df["Symbol"], df["DayChangePct"], color='green' if "UP" in category else 'red', alpha=0.7)
            plt.xlabel("Day Change %", fontsize=12)
            plt.ylabel("Symbol", fontsize=12)
            plt.title(f"{category.replace('_', ' ')} Stocks", fontsize=14)
            plt.grid(axis="x", linestyle="--", alpha=0.7)

            # Save plot
            file_path = os.path.join(self.output_dir, f"{category}_Bar.png")
            plt.savefig(file_path, bbox_inches="tight", format="png")
            plt.close()

            self.logger.info(f"Bar chart for {category} saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error generating bar chart for {category}: {e}")

    def plot_all_categories(self):
        """
        Generate bar charts for all categories in the stock lists.
        """
        try:
            for category, stocks in self.stock_lists.items():
                self.plot_bar_chart(category, stocks)
        except Exception as e:
            self.logger.error(f"Error generating all category bar charts: {e}")


import asyncio
import os
import logging
import json
from telegram import Bot

class TelegramChartSender:
    def __init__(self, config_path: str, logger=None):
        """
        Initializes the Telegram bot using the configuration file.

        Args:
            config_path (str): Path to the telegram_config.json file.
            logger (logging.Logger): Logger for logging messages.
        """
        # Load the configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        self.bot_token = config.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = config.get("TELEGRAM_CHAT_ID")
        self.group_id = config.get("TELEGRAM_GROUP_ID")

        if not self.bot_token or not self.chat_id:
            raise ValueError("Invalid configuration: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")

        self.bot = Bot(token=self.bot_token)
        self.logger = logger

    async def send_text_update(self, custom_text: str = None):
        """Sends a text message to the Telegram chat/channel."""
        if not custom_text:
            from datetime import datetime
            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            custom_text = f"Here is today's update @ {timestamp_str}"

        try:
            message = await self.bot.send_message(chat_id=self.chat_id, text=custom_text)
            if self.logger:
                self.logger.info(f"Sent text message: {custom_text}. Telegram response: {message}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error sending text message: {e}")

    async def send_charts(self, file_paths: list):
        """Sends a list of images to the Telegram chat/channel."""
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as chart_file:
                        response = await self.bot.send_photo(chat_id=self.chat_id, photo=chart_file)
                        if self.logger:
                            self.logger.info(f"Sent chart: {file_path}. Telegram response: {response}")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error sending chart {file_path}: {e}")
            else:
                if self.logger:
                    self.logger.warning(f"File not found: {file_path}")

    async def send_stock_lists(self, stock_lists: dict):
        """
        Sends stock lists (4%/10% up and down) as text messages to the Telegram chat/channel.

        Args:
            stock_lists (dict): A dictionary with categories as keys and lists of stock info as values.
        """
        for category, stock_list in stock_lists.items():
            try:
                message_text = f"**{category}**\n"
                if stock_list:
                    message_text += "\n".join(
                        [f"{stock['Symbol']}: Close={stock['Close']}, Change={stock['Day Change %']}%" for stock in stock_list]
                    )
                else:
                    message_text += "No stocks in this category."
                message = await self.bot.send_message(chat_id=self.chat_id, text=message_text)
                if self.logger:
                    self.logger.info(f"Sent stock list for {category}. Telegram response: {message}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error sending stock list for {category}: {e}")

    async def send_daily_update(self, file_paths: list, stock_lists: dict, custom_text: str = None):
        """
        Sends a text update, stock lists, and charts.

        Args:
            file_paths (list): List of file paths for charts.
            stock_lists (dict): Stock lists to send as text messages.
            custom_text (str): Optional text message to send.
        """
        try:
            # Send the main text update
            await self.send_text_update(custom_text=custom_text)
            
            # Send stock lists
            await self.send_stock_lists(stock_lists)
            
            # Send the charts
            await self.send_charts(file_paths)
            
            if self.logger:
                self.logger.info("Daily update sent successfully.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during daily update: {e}")


import os
import asyncio
import logging
from datetime import datetime
from pytz import timezone


# Define timezone
IST = timezone('Asia/Kolkata')

class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        """Override formatTime to ensure timestamps are in IST."""
        dt = datetime.fromtimestamp(record.created, IST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime('%Y-%m-%d %H:%M:%S')

def setup_logging(task_name):
    """Set up logging with IST timestamps."""
    # Define the logs directory
    logs_dir = os.path.join("/content", "logs")
    os.makedirs(logs_dir, exist_ok=True)  # Create the logs directory if it doesn't exist

    # Define the log file path
    current_date = datetime.now(IST).strftime("%Y-%m-%d")
    log_filename = os.path.join(logs_dir, f"logfile_{current_date}.log")

    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Define file and console handlers
    file_handler = logging.FileHandler(log_filename)
    console_handler = logging.StreamHandler()

    # Define the formatter with IST timestamps
    formatter = ISTFormatter(
        '%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    logger.info(f"Logger setup complete for {task_name}. Logfile: {log_filename}")

    return logger


import os
import asyncio
import logging
from datetime import datetime, time
from pytz import timezone


# Define timezone
IST = timezone('Asia/Kolkata')

class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        """Override formatTime to ensure timestamps are in IST."""
        dt = datetime.fromtimestamp(record.created, IST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime('%Y-%m-%d %H:%M:%S')

def setup_logging(task_name):
    """Set up logging with IST timestamps."""
    # Define the logs directory
    logs_dir = os.path.join("/content", "logs")
    os.makedirs(logs_dir, exist_ok=True)  # Create the logs directory if it doesn't exist

    # Define the log file path
    current_date = datetime.now(IST).strftime("%Y-%m-%d")
    log_filename = os.path.join(logs_dir, f"logfile_{current_date}.log")

    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Define file and console handlers
    file_handler = logging.FileHandler(log_filename)
    console_handler = logging.StreamHandler()

    # Define the formatter with IST timestamps
    formatter = ISTFormatter(
        '%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    logger.info(f"Logger setup complete for {task_name}. Logfile: {log_filename}")

    return logger

class TopGainersTopLosersPipeline:
    def __init__(self, config_path, symbol_file, output_dir, logger):
        """
        Initialize the pipeline for processing top gainers and losers.

        Args:
            config_path (str): Path to the telegram_config.json file.
            symbol_file (str): Path to the CSV file containing stock symbols.
            output_dir (str): Directory for saving outputs (data, plots, etc.).
            logger (logging.Logger): Logger instance for logging messages.
        """
        self.config_path = config_path
        self.symbol_file = symbol_file
        self.output_dir = output_dir
        self.logger = logger
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_greeting_message(self):
        """
        Generate a greeting message based on the current time.

        Returns:
            str: The appropriate greeting message.
        """
        now = datetime.now(IST).time()
        market_close_time = time(15, 30)  # 3:30 PM IST

        if now < market_close_time:
            return "Hello Group! Here is the interim Top Losers & Gainers List for Today"
        else:
            return "Hello Group! Here is the final Top Losers & Gainers List for Today"

    def run(self):
        """
        Execute the full pipeline: download data, analyze, plot, and send Telegram updates.
        """
        try:
            # Step 1: Data Download
            self.logger.info("Starting data download...")
            today = pd.Timestamp.today().strftime("%Y-%m-%d")
            dated_output_dir = os.path.join(self.output_dir, today)
            os.makedirs(dated_output_dir, exist_ok=True)

            downloader = DataDownloader(
                username="nileshiit",
                password="Hari@123om",
                symbol_file=self.symbol_file,
                output_dir=dated_output_dir,
                max_workers=5,
                logger=self.logger
            )
            downloader.download_and_save()

            # Step 2: Market Analysis
            self.logger.info("Starting market analysis...")
            data_file = os.path.join(dated_output_dir, f"TV_Daily_{today}.pkl")

            analysis = MarketAnalysis(input_dir=dated_output_dir, output_dir=dated_output_dir)
            analysis.load_data(data_file=data_file)
            analysis.process_data()

            # Step 3: Generate Bar Charts
            self.logger.info("Generating bar charts for gainers and losers...")
            plots_dir = os.path.join(dated_output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            market_plots = MarketPlots(stock_lists=analysis.stock_lists, output_dir=plots_dir)
            market_plots.plot_all_categories()

            # Collect generated chart paths
            chart_paths = [
                os.path.join(plots_dir, f"{category}_Bar.png")
                for category in ["5PCT_UP", "10PCT_UP", "5PCT_DOWN", "10PCT_DOWN"]
            ]

            # Step 4: Send Updates to Telegram
            self.logger.info("Sending updates to Telegram...")
            sender = TelegramChartSender(config_path=self.config_path, logger=self.logger)
            asyncio.run(sender.send_daily_update(
                file_paths=chart_paths,
                stock_lists=analysis.stock_lists,
                custom_text=self._generate_greeting_message()
            ))

            self.logger.info("Pipeline completed successfully.")

        except Exception as e:
            self.logger.error(f"Error during pipeline execution: {e}")


if __name__ == "__main__":
    logger = setup_logging("TopGainersTopLosersPipeline")

    # Configuration and file paths
    config_path = "/content/telegram_config.json"
    symbol_file = "/content/stock_list_25feb.csv"
    output_dir = "/content/TopGainersTopLosers"

    # Run the pipeline
    pipeline = TopGainersTopLosersPipeline(
        config_path=config_path,
        symbol_file=symbol_file,
        output_dir=output_dir,
        logger=logger
    )
    pipeline.run()

