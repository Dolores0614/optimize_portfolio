from flask import Flask, render_template, request
from datetime import datetime, timedelta
import yfinance as yf
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import requests

nltk.download('vader_lexicon')

app = Flask(__name__)

def fetch_stock_data(tickers):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=15*365)).strftime('%Y-%m-%d')
    all_data = {}
    valid_tickers = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
            if df.isna().sum() > 0:
                available_start_date = df.first_valid_index()
                if available_start_date is not None:
                    df = yf.download(ticker, start=available_start_date.strftime('%Y-%m-%d'), end=end_date)['Adj Close']
            if not df.empty:
                all_data[ticker] = df
                valid_tickers.append(ticker)
            else:
                print(f"Data for {ticker} is empty.")
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")

    if len(valid_tickers) == 0:
        raise ValueError("No valid tickers to optimize.")

    data_frame = pd.DataFrame(all_data)
    return data_frame.dropna(), valid_tickers


# Function to scrape news and analyze sentiment
def scrape_news_and_analyze_sentiment(tickers):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = {ticker: [] for ticker in tickers}

    for ticker in tickers:
        # Example URL, you may need to change it based on the website structure
        url = f"https://news.google.com/search?q={ticker}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        one_month_ago = datetime.today() - timedelta(days=30)
        
        for article in soup.find_all('article'):
            title_tag = article.find('h3') or article.find('h2')
            if title_tag:
                title = title_tag.get_text()
                date_tag = article.find('time')
                if date_tag and date_tag.has_attr('datetime'):
                    date_str = date_tag['datetime']
                    date = datetime.fromisoformat(date_str[:-1])
                    if date >= one_month_ago:
                        sentiment = analyzer.polarity_scores(title)
                        sentiment_scores[ticker].append(sentiment['compound'])

    # Calculate average sentiment score for each ticker
    average_sentiment = {ticker: np.mean(scores) if scores else 0 for ticker, scores in sentiment_scores.items()}
    return average_sentiment


# Function to adjust expected returns based on sentiment
def adjust_returns_with_sentiment(returns, sentiment_scores):
    adjusted_returns = returns.copy()
    for stock in adjusted_returns.index:
        sentiment = sentiment_scores.get(stock, 0)
        adjusted_returns[stock] += sentiment * 0.01  # Adjust this factor as needed
    return adjusted_returns

#support fuc
def calculate_var(df, confidence_level=0.95):
    returns = df.pct_change().dropna()
    var = returns.quantile(1 - confidence_level)
    return var

def calculate_sharpe_ratio(df):
    returns = df.pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized Sharpe Ratio
    return sharpe_ratio

import numpy as np
import cvxpy as cp
from pypfopt import EfficientFrontier, DiscreteAllocation, expected_returns, risk_models
from pypfopt.discrete_allocation import get_latest_prices

def optimize_portfolio(tickers, investment_amount, risk_preference):
    # Fetch historical stock prices
    df, valid_tickers = fetch_stock_data(tickers)
    
    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    
    # Calculate risk metrics for each stock
    var = calculate_var(df)
    sharpe_ratios = df.apply(calculate_sharpe_ratio)
    
    # Scrape news and analyze sentiment
    sentiment_scores = scrape_news_and_analyze_sentiment(valid_tickers)
    mu = adjust_returns_with_sentiment(mu, sentiment_scores)

    # Adjust returns based on risk preference
    if risk_preference == 'averse':
        mu -= var * 0.1  # Penalize high VaR stocks
    elif risk_preference == 'like':
        mu += sharpe_ratios * 0.1  # Favor high Sharpe Ratio stocks

    # Optimize for maximum Sharpe ratio
    ef = EfficientFrontier(mu, S)
    if risk_preference == 'averse':
        var_array = var.values  # Ensure var is a numpy array
        var_limit = np.percentile(var_array, 95)
        # Add constraint for VaR using cvxpy
        ef.add_constraint(lambda w: cp.sum(cp.multiply(var_array, w)) <= var_limit)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    # Calculate allocation based on investment amount
    latest_prices = get_latest_prices(df)
    
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=investment_amount)
    allocation, leftover = da.lp_portfolio()
    
    
    # Display portfolio performance
    performance = ef.portfolio_performance(verbose=True)
    
    return cleaned_weights, allocation, leftover, performance

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        tickers = request.form['tickers'].split(',')
        tickers = [ticker.strip() for ticker in tickers]
        investment_amount = float(request.form['investment_amount'])
        risk_preference_input = int(request.form['risk_preference'])

        risk_preference = 'neutral'
        if risk_preference_input == 1:
            risk_preference = 'averse'
        elif risk_preference_input == 3:
            risk_preference = 'like'

        try:
            weights, allocation, leftover, performance = optimize_portfolio(tickers, investment_amount, risk_preference)
            result = {
                'weights': weights,
                'allocation': allocation,
                'leftover': leftover,
                'performance': performance
            }
        except ValueError as e:
            result = {'error': str(e)}

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
