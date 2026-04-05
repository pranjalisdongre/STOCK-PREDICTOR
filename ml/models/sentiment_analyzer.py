import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from transformers import pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.sentiment_pipeline = None
        self.is_trained = False
    
    def load_pretrained_model(self):
        """Load pre-trained transformer model for sentiment analysis"""
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            print("✅ Pre-trained sentiment model loaded")
        except Exception as e:
            print(f"❌ Error loading pre-trained model: {e}")
            self.sentiment_pipeline = None
    
    def prepare_sentiment_data(self, news_df):
        """Prepare news data for sentiment analysis"""
        if news_df.empty:
            return None, None
        
        # Combine title and content for analysis
        texts = []
        sentiments = []
        
        for _, article in news_df.iterrows():
            text = f"{article['title']}. {article.get('description', '')}"
            texts.append(text)
            
            # Use existing sentiment as label if available
            if 'custom_sentiment' in article:
                sentiment = 1 if article['custom_sentiment'] > 0.1 else (
                    -1 if article['custom_sentiment'] < -0.1 else 0
                )
                sentiments.append(sentiment)
            else:
                sentiments.append(0)  # Neutral by default
        
        return texts, sentiments
    
    def extract_features(self, texts):
        """Extract features from text using TF-IDF"""
        if not texts:
            return np.array([])
        
        return self.tfidf_vectorizer.fit_transform(texts)
    
    def train_models(self, news_df):
        """Train sentiment classification models"""
        print("🤖 Training sentiment analysis models...")
        
        texts, sentiments = self.prepare_sentiment_data(news_df)
        
        if not texts or len(set(sentiments)) < 2:
            print("❌ Insufficient data for training")
            return
        
        # Extract features
        X = self.extract_features(texts)
        y = np.array(sentiments)
        
        # Train individual models
        for name, model in self.models.items():
            model.fit(X, y)
            print(f"✅ {name} trained")
        
        self.is_trained = True
    
    def analyze_sentiment_advanced(self, text):
        """Analyze sentiment using multiple approaches"""
        if not text or len(text.strip()) < 10:
            return {'sentiment': 0, 'confidence': 0, 'method': 'insufficient_text'}
        
        results = {}
        
        # Method 1: Pre-trained transformer
        if self.sentiment_pipeline:
            try:
                transformer_result = self.sentiment_pipeline(text[:512])[0]  # Limit length
                label = transformer_result['label']
                score = transformer_result['score']
                
                # Map to numeric sentiment
                if label == 'positive':
                    results['transformer'] = score
                elif label == 'negative':
                    results['transformer'] = -score
                else:
                    results['transformer'] = 0
            except Exception as e:
                print(f"Transformer analysis error: {e}")
                results['transformer'] = 0
        
        # Method 2: Trained classifiers
        if self.is_trained:
            features = self.tfidf_vectorizer.transform([text])
            
            for name, model in self.models.items():
                try:
                    prediction = model.predict(features)[0]
                    results[name] = prediction
                except Exception as e:
                    print(f"{name} prediction error: {e}")
                    results[name] = 0
        
        # Combine results
        if results:
            final_sentiment = np.mean(list(results.values()))
            confidence = np.std(list(results.values()))
        else:
            final_sentiment = 0
            confidence = 0
        
        return {
            'sentiment': final_sentiment,
            'confidence': 1 - confidence,  # Lower std dev = higher confidence
            'methods_used': list(results.keys()),
            'individual_scores': results
        }
    
    def get_market_sentiment(self, news_articles):
        """Calculate overall market sentiment from multiple news articles"""
        if not news_articles:
            return {'overall_sentiment': 0, 'confidence': 0, 'article_count': 0}
        
        sentiments = []
        confidences = []
        
        for article in news_articles:
            text = f"{article['title']}. {article.get('description', '')}"
            analysis = self.analyze_sentiment_advanced(text)
            
            sentiments.append(analysis['sentiment'])
            confidences.append(analysis['confidence'])
        
        if sentiments:
            # Weight by confidence
            weighted_sentiment = np.average(sentiments, weights=confidences)
            avg_confidence = np.mean(confidences)
            
            return {
                'overall_sentiment': weighted_sentiment,
                'confidence': avg_confidence,
                'article_count': len(news_articles),
                'positive_articles': sum(1 for s in sentiments if s > 0.1),
                'negative_articles': sum(1 for s in sentiments if s < -0.1),
                'neutral_articles': sum(1 for s in sentiments if -0.1 <= s <= 0.1)
            }
        
        return {'overall_sentiment': 0, 'confidence': 0, 'article_count': 0}

# Test function
def test_sentiment_analyzer():
    """Test the sentiment analyzer"""
    print("🧪 Testing Sentiment Analyzer...")
    
    analyzer = AdvancedSentimentAnalyzer()
    analyzer.load_pretrained_model()
    
    # Sample news articles
    sample_articles = [
        {
            'title': 'Company reports record profits and strong growth',
            'description': 'The company exceeded analyst expectations with outstanding quarterly results.'
        },
        {
            'title': 'Market downturn affects tech stocks',
            'description': 'Technology companies face challenges due to economic conditions.'
        },
        {
            'title': 'New product launch receives positive reviews',
            'description': 'The innovative product is expected to drive future revenue growth.'
        }
    ]
    
    # Test individual sentiment analysis
    for i, article in enumerate(sample_articles):
        text = f"{article['title']}. {article['description']}"
        result = analyzer.analyze_sentiment_advanced(text)
        print(f"Article {i+1}: Sentiment={result['sentiment']:.3f}, Confidence={result['confidence']:.3f}")
    
    # Test market sentiment
    market_sentiment = analyzer.get_market_sentiment(sample_articles)
    print(f"📊 Market Sentiment: {market_sentiment}")
    
    return analyzer

if __name__ == "__main__":
    test_sentiment_analyzer()