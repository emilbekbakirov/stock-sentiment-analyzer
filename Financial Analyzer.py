import yfinance as yf
from transformers import pipeline
pipe = pipeline("text-classification", model="ProsusAI/finbert")
import numpy as np
import matplotlib.pyplot as plt

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="/Users/emilbekbakirov/Desktop/.env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_news(ticker):
    stock = yf.Ticker(ticker)
    news = stock.news
    return news

while True:
    list_of_tickers_str = input("Enter a stock tickers you want to compare (or q to quit): ").upper()
    list_of_tickers = [t.strip() for t in list_of_tickers_str.split(",")]    
    if list_of_tickers_str == "Q":
        break
    
    list_of_controvercy_scores = []
    list_of_strength_values = []

    prompt = "Analyze these companies as a financial analyst: \n"
    for ticker in list_of_tickers:
        
# to calc the stock sentiment stats
        neg_score = 0
        neut_score = 0
        pos_score = 0
        neg_num = 0
        neut_num = 0
        pos_num = 0
        articles = get_news(ticker)
        for article in articles:
            content = article["content"]
            evaluation = pipe(content["title"])
            label = evaluation[0]["label"]
            score = evaluation[0]["score"]
            if label == "negative":
                neg_num += 1
                neg_score += score
            elif label == "neutral":
                neut_num += 1
                neut_score += score
            else:
                pos_num += 1
                pos_score += score

        average_negative_score = neg_score / neg_num if neg_num > 0 else 0
        average_neutral_score = neut_score / neut_num if neut_num > 0 else 0
        average_positive_score = pos_score / pos_num if pos_num > 0 else 0
# for the plot of controvercy and strength
        strength_value = average_positive_score - average_negative_score
        controvercy_score = average_positive_score + average_negative_score
        list_of_controvercy_scores.append(controvercy_score)
        list_of_strength_values.append(strength_value)

        stock_sentiment = (f"\nThe stock market sentiment for {ticker}: \n Positive: {pos_num} articles, average score: {round(average_positive_score, 2)}, \n Negative: {neg_num} articles, average score: {round(average_negative_score, 2)}, \n  Neutral:  {neut_num} articles, average score: {round(average_neutral_score, 2)}\n")
        prompt += stock_sentiment
# to calc the key ratios        
        stock = yf.Ticker(ticker)
        info = stock.info
        pe_ratio = info.get("trailingPE", 0)
        trailing_eps = info.get("trailingEps", 0)
        revenueGrowth = info.get("revenueGrowth", 0)
        marketCap = info.get("marketCap", 0)
        summary = f"The key stock ratios for {ticker}: \n PE ratio : {pe_ratio},\n Trailing EPS : {trailing_eps},\n Revenue Growth : {revenueGrowth},\n Market Capitalization : {marketCap}\n"
        prompt += summary
        prompt += ("-----")

    
    prompt += """
    Write a comprehensive financial analysis with the following structure:

    1. MARKET SENTIMENT OVERVIEW
    Analyze the sentiment scores for each company. What does the market feel about each stock right now?

    2. FINANCIAL HEALTH COMPARISON  
    Compare the key ratios. Which company is cheapest on valuation? Which is most profitable? Which is growing fastest?

    3. RISK ASSESSMENT
    Based on the controversy scores and financial ratios, which company carries the most risk?

    4. INVESTMENT VERDICT
    Give a clear verdict for each company: Buy, Hold, or Avoid — and explain why in 2 sentences each.
    """

    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{
        "role": "user",
        "content": f"{prompt}"
    }])

    print(response.choices[0].message.content)

    
    plt.figure()
    x = np.arange(len(list_of_tickers))
    width = 0.35

    plt.bar(x - width/2, list_of_strength_values, width, label="Sentiment Strength", color="green")
    plt.bar(x + width/2, list_of_controvercy_scores, width, label="Controversy Level", color="red")

    plt.xticks(x, list_of_tickers)
    plt.legend()
    plt.title("Sentiment Analysis by Company")
    plt.show()