import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

BASE_URL = "https://kathmandupost.com"

def scrape_kathmandu_post(pages=20):
    articles = []

    for page in range(1, pages + 1):
        url = f"{BASE_URL}/news?page={page}"
        print(f"Scraping Kathmandu Post page {page}...")

        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            cards = soup.find_all("article")

            for card in cards:
                link_tag = card.find("a")
                if not link_tag:
                    continue

                link = BASE_URL + link_tag.get("href")

                try:
                    article_r = requests.get(link, timeout=10)
                    article_soup = BeautifulSoup(article_r.text, "html.parser")
                    paragraphs = article_soup.find_all("p")
                    text = " ".join([p.text.strip() for p in paragraphs])

                    if len(text) > 300:
                        title = article_soup.find("h1").text.strip()
                        articles.append([title, text, "real"])

                except:
                    continue

            time.sleep(random.uniform(1, 2))

        except:
            continue

    df = pd.DataFrame(articles, columns=["title", "text", "label"])
    df.to_csv("dataset/kathmandu_post_raw.csv", index=False)
    print("Saved Kathmandu Post dataset!")

scrape_kathmandu_post()
