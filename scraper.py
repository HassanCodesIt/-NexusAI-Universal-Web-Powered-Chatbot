import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
import random

# --- User-Agent Rotation ---
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
]

def extract_text_from_soup(soup):
    """
    Extracts text from a BeautifulSoup object from common tags.
    """
    texts = []
    # Extract from primary content tags
    for tag in ['h1', 'h2', 'h3', 'p', 'li', 'span', 'div']:
        elements = soup.find_all(tag)
        for element in elements:
            # Avoid extracting text from script, style, and navigation elements
            if element.find_parent(['script', 'style', 'nav', 'footer', 'header']):
                continue
            text = element.get_text(separator=' ', strip=True)
            if text:
                texts.append(text)
    
    # Simple deduplication while preserving order
    unique_texts = []
    seen = set()
    for text in texts:
        if text not in seen:
            unique_texts.append(text)
            seen.add(text)
            
    return "\n".join(unique_texts)

def scrape_static_content(url):
    """
    Scrapes a static website using requests and BeautifulSoup.
    """
    try:
        headers = {'User-Agent': random.choice(user_agents)}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return extract_text_from_soup(soup)
    except requests.exceptions.RequestException as e:
        print(f"Error scraping static content from {url}: {e}")
        return None

def scrape_dynamic_content(url):
    """
    Scrapes a dynamic website using Selenium with improved stability and error handling.
    """
    driver = None  # Ensure driver is defined for the finally block
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
        chrome_options.add_argument("user-agent=" + random.choice(user_agents))
        
        print("Initializing Chrome driver for dynamic scraping...")
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        print(f"Driver initialized. Getting URL: {url}")
        
        driver.get(url)
        # A slightly longer wait can help with very heavy JS sites
        time.sleep(7)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        print(f"Successfully scraped content from {url}")
        return extract_text_from_soup(soup)
    except Exception as e:
        import traceback
        print(f"An exception occurred in scrape_dynamic_content: {e}")
        print(traceback.format_exc())
        return "Failed to scrape dynamic content due to an error."
    finally:
        if driver:
            print("Quitting Chrome driver.")
            driver.quit()

def scrape_site(url):
    """
    Determines the best method to scrape a site and returns the content.
    Strategy: Try static first. If content is too sparse, fall back to dynamic.
    """
    print(f"Scraping URL: {url}")
    # First, attempt static scraping
    static_content = scrape_static_content(url)
    
    # Heuristic to decide if static content is sufficient
    if static_content and len(static_content) > 500: # 500 characters as a threshold
        print("Static scraping successful with sufficient content.")
        return static_content
    
    print("Static content is sparse or failed, attempting dynamic scraping with Selenium...")
    dynamic_content = scrape_dynamic_content(url)
    
    return dynamic_content

def scrape_and_return(url):
    """
    Main wrapper function to scrape a URL and return a structured dictionary.
    """
    content = scrape_site(url)
    if content:
        return {
            "url": url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "content": content
        }
    else:
        return {
            "url": url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "content": "Failed to retrieve content from the URL."
        } 