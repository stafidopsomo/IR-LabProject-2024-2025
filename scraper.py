import requests
from bs4 import BeautifulSoup
import json

def scrape_wikipedia_articles(url, num_articles=5):
    base_url = "https://en.wikipedia.org"
    articles = []
    visited_urls = set()

    def get_links(page_url):
        """Extract all Wikipedia article links from a given page."""
        response = requests.get(page_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []
        for link in soup.find_all("a", href=True):
            href = link['href']
            if href.startswith("/wiki/") and ":" not in href:  # Avoid non-article links
                full_url = base_url + href
                if full_url not in visited_urls:
                    links.append(full_url)
        return links

    # Start with the main Wikipedia page
    to_visit = [url]
    while to_visit and len(articles) < num_articles:
        current_url = to_visit.pop(0)
        visited_urls.add(current_url)
        try:
            response = requests.get(current_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find("h1").text.strip()
            content = "\n".join([p.text for p in soup.find_all("p")])
            if content:
                articles.append({"title": title, "url": current_url, "content": content})
                print(f"Scraped: {title}")
            to_visit.extend(get_links(current_url))
        except Exception as e:
            print(f"Failed to scrape {current_url}: {e}")

    return articles

# Scrape articles and save to JSON
url = "https://en.wikipedia.org/wiki/Main_Page"
articles = scrape_wikipedia_articles(url, num_articles=5)

# Save to JSON file
with open("wikipedia_articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=4)

print("Scraping complete. Data saved to wikipedia_articles.json.")
