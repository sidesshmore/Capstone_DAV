import streamlit as st
import pandas as pd
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def convert_to_sample_link(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    pid = query_params.get('pid', [''])[0]
    lid = query_params.get('lid', [''])[0]
    marketplace = query_params.get('marketplace', [''])[0]
    q = query_params.get('q', [''])[0]
    store = query_params.get('store', [''])[0]
    spotlightTagId = query_params.get('spotlightTagId', [''])[0]
    srno = query_params.get('srno', [''])[0]

    sample_link = f"https://www.flipkart.com/{q}/product-reviews/{pid}?pid={pid}&lid={lid}&marketplace={marketplace}&q={q}&store={store}&spotlightTagId={spotlightTagId}&srno={srno}"
    
    return sample_link


def scrape_reviews(url):
    reviews = []
    page = 1

    while True:
        page_url = url + f"&page={page}"
        response = requests.get(page_url)
        soup = BeautifulSoup(response.content, "html.parser")
        review_blocks = soup.find_all('div', {'class': '_27M-vq'})

        if not review_blocks:
            break

        for block in review_blocks:
            rating_elem = block.find('div', {'class': '_3LWZlK'})
            review_elem = block.find('p', {'class': '_2-N8zT'})
            sum_elem = block.find('div', {'class': 't-ZTKy'})
            name_elem = block.find_all('p', {'class': '_2sc7ZR'})[0]
            date_elem = block.find_all('p', {'class': '_2sc7ZR'})[1]
            location_elem = block.find('p', {'class': '_2mcZGG'})

            if rating_elem and review_elem and name_elem and date_elem:
                review = {
                    'Rating': rating_elem.text,
                    'Review': review_elem.text,
                    'Name': name_elem.text.strip(),
                    'Date': date_elem.text.strip(),
                    'Review Description': sum_elem.text.strip(),
                    'Location': location_elem.text
                }
                reviews.append(review)

        page += 1

    return reviews


def analyze_sentiment(review):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(review)
    
    if sentiment_score['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Streamlit app
def main():
    st.title("Flipkart Product Reviews Analysis")


    product_url = st.text_input("Paste the Flipkart product URL here:")

    if product_url:
        sample_link = convert_to_sample_link(product_url)


        reviews = scrape_reviews(sample_link)


        st.subheader("Scraped Reviews:")
        st.write(pd.DataFrame(reviews))


        analyze_reviews(reviews)

def analyze_reviews(reviews):

    ratings = [float(review['Rating']) for review in reviews if review['Rating'].isdigit()]
    if ratings:
        average_rating = sum(ratings) / len(ratings)
        st.write(f"Average Rating: {average_rating:.2f}")
    else:
        st.write("No ratings found")


    review_sentiments = [analyze_sentiment(review['Review']) for review in reviews]
    description_sentiments = [analyze_sentiment(review['Review Description']) for review in reviews]


    st.subheader("Sentiment Analysis Results:")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Sentiment Analysis for Reviews
    review_sentiment_counts = pd.Series(review_sentiments).value_counts()
    axes[0].bar(review_sentiment_counts.index, review_sentiment_counts.values)
    axes[0].set_title("Sentiment Analysis for Reviews")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Count")
    axes[0].grid(True)

    # Sentiment Analysis for Review Descriptions
    description_sentiment_counts = pd.Series(description_sentiments).value_counts()
    axes[1].bar(description_sentiment_counts.index, description_sentiment_counts.values)
    axes[1].set_title("Sentiment Analysis for Review Descriptions")
    axes[1].set_xlabel("Sentiment")
    axes[1].set_ylabel("Count")
    axes[1].grid(True)

    # Pie chart for Sentiment Analysis of Review Descriptions
    sentiment_colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'lightcoral'}
    description_sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=[sentiment_colors[s] for s in description_sentiment_counts.index], ax=axes[2])
    axes[2].set_title("Sentiment Analysis for Review Descriptions")
    axes[2].set_ylabel("")
    axes[2].legend(description_sentiment_counts.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    axes[2].grid(True)

    st.pyplot(fig)


    st.subheader("Top 5 Locations:")
    location_counts = pd.Series([review['Location'] for review in reviews]).value_counts().head(5)
    st.write(location_counts)

if __name__ == "__main__":
    main()
