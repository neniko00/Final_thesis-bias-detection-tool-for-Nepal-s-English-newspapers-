import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

BASE_URL = "https://southasiacheck.org"

def scrape_south_asia_check(pages=15):
    articles = []

    for page in range(1, pages + 1):
        url = f"{BASE_URL}/fact-check/page/{page}/"
        print(f"Scraping South Asia Check page {page}...")

        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            posts = soup.find_all("h2", class_="entry-title")

            for post in posts:
                link = post.find("a")["href"]

                try:
                    article_r = requests.get(link, timeout=10)
                    article_soup = BeautifulSoup(article_r.text, "html.parser")
                    title = article_soup.find("h1").text.strip()
                    paragraphs = article_soup.find_all("p")
                    text = " ".join([p.text.strip() for p in paragraphs])

                    if len(text) > 300:
                        articles.append([title, text, "real"])

                except:
                    continue

            time.sleep(random.uniform(1, 2))

        except:
            continue

    df = pd.DataFrame(articles, columns=["title", "text", "label"])
    df.to_csv("dataset/southasiacheck_raw.csv", index=False)
    print("Saved South Asia Check dataset!")

scrape_south_asia_check()
