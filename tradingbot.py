import os
from datetime import datetime

from alpaca_trade_api import REST
from finbert_utils import estimate_sentiment
from joblib import Memory
from lumibot.backtesting import PolygonDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.strategies.strategy import Strategy
from timedelta import Timedelta

BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY": os.environ.get("APCA_API_KEY_ID"),
    "API_SECRET": os.environ.get("APCA_API_SECRET_KEY"),
    "PAPER": True,
}

os.makedirs("cache/tradingbot", exist_ok=True)
memory = Memory(location="cache/tradingbot", verbose=0)

alpaca_api = REST(
    base_url=BASE_URL, key_id=ALPACA_CREDS["API_KEY"], secret_key=ALPACA_CREDS["API_SECRET"]
)


@memory.cache
def get_news(symbol: str, start: str, end: str):
    global alpaca_api
    return alpaca_api.get_news(symbol=symbol, start=start, end=end)


class _MlTrader(Strategy):
    def initialize(self, symbol: str = "SPY", cash_at_risk: float = 0.5):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk


    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return (cash, last_price, quantity)

    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return (today.strftime("%Y-%m-%d"), three_days_prior.strftime("%Y-%m-%d"))

    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = get_news(symbol=self.symbol, start=three_days_prior, end=today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return (probability, sentiment)

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        if cash > last_price:
            if sentiment == "positive" and probability > 0.999:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.2,
                    stop_loss_price=last_price * 0.95,
                )
                self.submit_order(order)
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > 0.999:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.8,
                    stop_loss_price=last_price * 1.05,
                )
                self.submit_order(order)
                self.last_trade = "sell"


START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2023, 12, 31)
BROKER = Alpaca(ALPACA_CREDS)
STRATEGY = _MlTrader(
    name="mlstrat", broker=BROKER, parameters={"symbol": "SPY", "cash_at_risk": 0.5}
)

STRATEGY.backtest(
    PolygonDataBacktesting,
    START_DATE,
    END_DATE,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5},
    polygon_api_key=os.environ.get("POLYGON_API_KEY"),
    polygon_has_paid_subscription=True,
)