import requests
import pandas as pd
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newspaper import Article
import time
from config import Config

class NewsSentimentAnalyzer:
    def __init__(self):
        self.api_key = Config.NEWS_API_KEY
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.news_cache = {}
        
    def fetch_news(self, symbol, days=7, language='en'):
        """Fetch recent news for a stock symbol"""
        if self.api_key == 'your_news_api_key_here':
            print("⚠️  Please set up your NewsAPI key in config.py")
            return self.get_sample_news(symbol)
        
        url = "https://newsapi.org/v2/everything"
        
        params = {
            'q': f"{symbol} stock OR {symbol} shares OR {symbol} earnings",
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'apiKey': self.api_key,
            'language': language,
            'pageSize': 20
        }
        
        try:
            print(f"📰 Fetching news for {symbol}...")
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"❌ News API error: {response.status_code}")
                return self.get_sample_news(symbol)
                
            articles = response.json().get('articles', [])
            print(f"✅ Found {len(articles)} articles for {symbol}")
            
            return self.process_articles(articles, symbol)
            
        except Exception as e:
            print(f"❌ Error fetching news: {e}")
            return self.get_sample_news(symbol)
    
    def process_articles(self, articles, symbol):
        """Process articles and calculate sentiment scores"""
        sentiment_data = []
        
        for article in articles:
            try:
                title = article['title']
                description = article['description'] or ''
                content = f"{title}. {description}"
                
                # Skip if no meaningful content
                if len(content) < 20:
                    continue
                
                # Calculate multiple sentiment scores
                blob = TextBlob(content)
                textblob_sentiment = blob.sentiment.polarity
                textblob_subjectivity = blob.sentiment.subjectivity
                
                vader_scores = self.vader_analyzer.polarity_scores(content)
                
                # Calculate custom sentiment score (weighted average)
                custom_score = (
                    textblob_sentiment * 0.3 +
                    vader_scores['compound'] * 0.7
                )
                
                sentiment_data.append({
                    'symbol': symbol,
                    'title': title,
                    'source': article['source']['name'],
                    'published_at': article['publishedAt'],
                    'url': article['url'],
                    'textblob_sentiment': textblob_sentiment,
                    'textblob_subjectivity': textblob_subjectivity,
                    'vader_positive': vader_scores['pos'],
                    'vader_negative': vader_scores['neg'],
                    'vader_neutral': vader_scores['neu'],
                    'vader_compound': vader_scores['compound'],
                    'custom_sentiment': custom_score,
                    'content_length': len(content)
                })
                
            except Exception as e:
                print(f"Error processing article: {e}")
                continue
        
        return pd.DataFrame(sentiment_data)
    
    def get_sample_news(self, symbol):
        """Generate sample news data for testing"""
        sample_articles = [
            {
                'title': f"{symbol} Reports Strong Quarterly Earnings",
                'description': f"{symbol} exceeds market expectations with record profits.",
                'source': {'name': 'Financial News'},
                'publishedAt': datetime.now().isoformat(),
                'url': 'https://example.com/1'
            },
            {
                'title': f"Analysts Upgrade {symbol} Price Target",
                'description': f"Multiple analysts raise price targets for {symbol} stock.",
                'source': {'name': 'Market Watch'},
                'publishedAt': (datetime.now() - timedelta(hours=2)).isoformat(),
                'url': 'https://example.com/2'
            }
        ]
        
        return self.process_articles(sample_articles, symbol)
    
    def calculate_daily_sentiment(self, symbol):
        """Calculate aggregated daily sentiment"""
        news_df = self.fetch_news(symbol)
        
        if not news_df.empty:
            # Convert published_at to datetime
            news_df['published_at'] = pd.to_datetime(news_df['published_at'])
            
            # Calculate recency weight (more recent = higher weight)
            news_df['hours_ago'] = (datetime.now() - news_df['published_at']).dt.total_seconds() / 3600
            news_df['recency_weight'] = np.exp(-news_df['hours_ago'] / 24)  # Exponential decay
            
            # Calculate weighted sentiment
            weighted_sentiment = np.average(
                news_df['custom_sentiment'], 
                weights=news_df['recency_weight']
            )
            
            # Calculate sentiment strength
            sentiment_strength = np.average(
                news_df['textblob_subjectivity'],
                weights=news_df['recency_weight']
            )
            
            return {
                'symbol': symbol,
                'sentiment_score': weighted_sentiment,
                'sentiment_strength': sentiment_strength,
                'article_count': len(news_df),
                'positive_articles': len(news_df[news_df['custom_sentiment'] > 0.1]),
                'negative_articles': len(news_df[news_df['custom_sentiment'] < -0.1]),
                'timestamp': datetime.now()
            }
        
        return None
    
    def get_sentiment_timeseries(self, symbols, days=30):
        """Get sentiment time series for multiple symbols"""
        sentiment_series = {}
        
        for symbol in symbols:
            daily_sentiments = []
            
            for day in range(days):
                date = datetime.now() - timedelta(days=day)
                # In a real implementation, you'd fetch historical news
                # For now, we'll simulate some data
                sentiment = self.calculate_daily_sentiment(symbol)
                if sentiment:
                    daily_sentiments.append(sentiment)
            
            if daily_sentiments:
                sentiment_series[symbol] = pd.DataFrame(daily_sentiments)
        
        return sentiment_series

# Test function
def test_news_collector():
    """Test the news collector"""
    analyzer = NewsSentimentAnalyzer()
    
    print("Testing news sentiment analysis...")
    sentiment = analyzer.calculate_daily_sentiment('AAPL')
    
    if sentiment:
        print(f"Sentiment Analysis for AAPL:")
        print(f"Score: {sentiment['sentiment_score']:.3f}")
        print(f"Strength: {sentiment['sentiment_strength']:.3f}")
        print(f"Articles: {sentiment['article_count']}")
        print(f"Positive: {sentiment['positive_articles']}")
        print(f"Negative: {sentiment['negative_articles']}")
    
    return analyzer

if __name__ == "__main__":
    test_news_collector()