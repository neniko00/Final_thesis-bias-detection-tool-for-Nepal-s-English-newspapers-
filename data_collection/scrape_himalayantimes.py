import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

BASE_URL = "https://thehimalayantimes.com"

def scrape_himalayan_times(pages=20):
    articles = []

    for page in range(1, pages + 1):
        url = f"{BASE_URL}/latest-news?page={page}"
        print(f"Scraping Himalayan Times page {page}...")

        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            news_cards = soup.find_all("div", class_="latest-news")

            for card in news_cards:
                title = card.find("h2").text.strip()
                link = BASE_URL + card.find("a")["href"]

                # Fetch article text
                try:
                    article_r = requests.get(link, timeout=10)
                    article_soup = BeautifulSoup(article_r.text, "html.parser")

                    paragraphs = article_soup.find_all("p")
                    text = " ".join([p.text.strip() for p in paragraphs])

                    if len(text) > 300:  # minimum article length
                        articles.append([title, text, "real"])
                except:
                    continue

            time.sleep(random.uniform(1, 2))

        except:
            print("Error loading page.")
            continue

    df = pd.DataFrame(articles, columns=["title", "text", "label"])
    df.to_csv("dataset/himalayan_times_raw.csv", index=False)
    print("Saved Himalayan Times dataset!")

scrape_himalayan_times()
