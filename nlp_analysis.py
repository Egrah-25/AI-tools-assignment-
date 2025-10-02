import spacy
from spacy import displacy
import random

# Load spaCy model
print("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please download the model first: python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Sample Amazon product reviews
reviews = [
    "I bought the new iPhone 15 from Apple and it's absolutely amazing! The camera quality is outstanding.",
    "This Samsung Galaxy phone stopped working after just 2 weeks. Very disappointed with the quality.",
    "The Sony headphones have incredible sound quality and the battery lasts forever.",
    "Avoid this HP laptop at all costs! It constantly freezes and the customer service is terrible.",
    "My new MacBook Pro from Apple works perfectly for programming and design work.",
    "The Microsoft Surface tablet is good but the keyboard cover is overpriced.",
    "Bose QuietComfort headphones are worth every penny for noise cancellation.",
    "This Dell computer crashed on the first day and lost all my important files."
]

# Sentiment analysis using rule-based approach
def analyze_sentiment(text):
    """
    Simple rule-based sentiment analysis
    """
    positive_words = ['amazing', 'outstanding', 'incredible', 'perfectly', 'good', 'great', 'excellent', 'wonderful', 'fantastic', 'worth']
    negative_words = ['disappointed', 'terrible', 'avoid', 'crashed', 'freezes', 'overpriced', 'stopped']
    
    text_lower = text.lower()
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

print("=== Named Entity Recognition and Sentiment Analysis ===\n")

# Process each review
for i, review in enumerate(reviews, 1):
    print(f"Review {i}: {review}")
    
    # Process with spaCy
    doc = nlp(review)
    
    # Extract entities (focusing on products and brands)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Filter for relevant entities (ORG, PRODUCT, etc.)
    relevant_entities = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "GPE"]:  # Organizations, Products, Countries/Cities
            relevant_entities.append((ent.text, ent.label_))
    
    # Analyze sentiment
    sentiment = analyze_sentiment(review)
    
    print(f"Sentiment: {sentiment}")
    print(f"Extracted Entities: {relevant_entities}")
    
    # Display entities using spaCy's visualizer for the first review
    if i == 1:
        print("\n=== Entity Visualization (First Review) ===")
        displacy.render(doc, style="ent", jupyter=True)
    
    print("-" * 80)

# Additional analysis: Most mentioned brands
print("\n=== Brand Analysis ===")
brand_mentions = {}
brand_sentiments = {}

for review in reviews:
    doc = nlp(review)
    sentiment = analyze_sentiment(review)
    
    for ent in doc.ents:
        if ent.label_ == "ORG":  # Organization (brands)
            brand = ent.text
            if brand not in brand_mentions:
                brand_mentions[brand] = 0
                brand_sentiments[brand] = []
            brand_mentions[brand] += 1
            brand_sentiments[brand].append(sentiment)

print("Brand mentions and sentiment distribution:")
for brand, count in brand_mentions.items():
    sentiments = brand_sentiments[brand]
    print(f"{brand}: {count} mentions - Sentiments: {sentiments}")
