import os
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, send_from_directory
from huggingface_hub import InferenceClient
import datetime
import requests
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from urllib.parse import urlparse
import json
import faiss
from sentence_transformers import SentenceTransformer
import re
import numpy as np
import traceback

# Import the generalized scraper
from scraper import scrape_and_return

# Load environment variables from .env
load_dotenv()

# --- API Key Rotation Setup ---
def get_all_api_keys():
    keys = []
    i = 0
    while True:
        key_name = 'apikey' if i == 0 else f'apikey{i}'
        key = os.getenv(key_name)
        if key:
            keys.append(key)
            i += 1
        else:
            break
    return keys

API_KEYS = get_all_api_keys()
if not API_KEYS:
    raise ValueError('No API keys found in .env file.')

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

# Helper to try all keys in order for LLM calls
def call_llm_with_key_rotation(create_func, *args, **kwargs):
    last_exception = None
    for key in API_KEYS:
        try:
            client = InferenceClient(provider="nebius", api_key=key)
            return create_func(client, *args, **kwargs)
        except Exception as e:
            # Check for quota or auth errors (customize as needed)
            if any(x in str(e).lower() for x in ["quota", "limit", "unauthorized", "forbidden", "rate limit", "429"]):
                print(f"API key failed due to quota/auth: {e}. Trying next key...")
                last_exception = e
                continue
            else:
                raise
    raise RuntimeError(f"All API keys failed. Last error: {last_exception}")

# Initialize the markdown converter
md = MarkdownIt()

app = Flask(__name__)

# --- Trusted Sources Loading ---
TRUSTED_SOURCES_FILE = "trusted_sources.txt"
tier1_domains = set()
tier2_domains = set()

def extract_domain(url):
    # Extracts the domain from a URL (e.g., bbc.com from https://www.bbc.com/sport)
    match = re.search(r"([a-zA-Z0-9\-]+\.[a-zA-Z]{2,})(/|$)", url)
    return match.group(1).replace('www.', '') if match else None

def load_trusted_sources():
    current_tier = None
    with open(TRUSTED_SOURCES_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith('tier 1'):
                current_tier = 1
            elif line.lower().startswith('tier 2'):
                current_tier = 2
            elif line.startswith('- '):
                # Extract domain(s) from the line
                urls = re.findall(r'\(([^)]+)\)', line)
                for url in urls:
                    for u in url.split(','):
                        domain = extract_domain(u.strip())
                        if domain:
                            if current_tier == 1:
                                tier1_domains.add(domain)
                            elif current_tier == 2:
                                tier2_domains.add(domain)

load_trusted_sources()
# --- End Trusted Sources Loading ---

# --- FAISS Vector Cache Setup (Optimized for Low Memory) ---
embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
# After initializing embedding_model, auto-detect embedding dimension
sample_emb = embedding_model.encode(["test"]).astype('float32')
EMBED_DIM = sample_emb.shape[1]
faiss_index = faiss.IndexFlatL2(EMBED_DIM)
vector_cache = []  # Each entry: (embedding, doc_id, user_message, answer_html)
MAX_CACHE_SIZE = 100
# --- End FAISS Vector Cache Setup ---

def search_for_urls(query, max_results=3):
    """
    Searches DuckDuckGo for a query and returns the URLs of the top results, prioritizing trusted sources.
    """
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" }
        response = requests.get(search_url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        results = soup.select(".result__url")
        urls = []
        for result in results:
            url = result.get_text(strip=True)
            full_url = f"https://{url}" if not url.startswith("http") else url
            urls.append(full_url)
        # --- Prioritize trusted sources ---
        def get_tier(url):
            domain = extract_domain(url)
            if domain in tier1_domains:
                return 1
            elif domain in tier2_domains:
                return 2
            else:
                return 3
        urls_sorted = sorted(urls, key=get_tier)
        # Only return up to max_results
        return urls_sorted[:max_results]
    except requests.exceptions.RequestException as e:
        print(f"Error searching for URLs: {e}")
        return None

def structure_data_with_llm(content, query, urls):
    """
    Uses an LLM to structure raw scraped text from multiple sources into a clean, 
    readable summary in markdown, and includes the source URLs as formatted hyperlinks.
    """
    print("Structuring scraped data with LLM...")
    system_prompt = (
        "You are an expert at synthesizing information from multiple sources. A user has asked a question, and raw text from several webpages has been provided. "
        "Your task is to analyze all the text and provide a single, coherent, and concise answer to the user's question. "
        "Format the answer nicely using markdown (headings, bold text, bullet points). "
        "IMPORTANT: After providing the complete answer, you MUST add a 'Sources' section at the very end of your response. Nothing should come after it. "
        "In this 'Sources' section, create a markdown hyperlink for each source URL provided in a bulleted list. "
        "The link text MUST be the main domain name only (e.g., for 'https://www.bbc.com/sports/article', use 'bbc'; for 'https://www.cnn.com/...' use 'cnn'). "
        "The link destination must be the original URL. For example: '* [bbc](https://www.bbc.com/sports/article)'. "
        "If the combined text does not contain a clear answer, state that you could not find a clear answer, but still provide the sources you checked at the end."
    )
    
    source_list = '\n'.join(f"- {url}" for url in urls)
    user_prompt = f"The user's question was: '{query}'.\n\nThe source URLs are:\n{source_list}\n\nHere is the combined raw text from the webpages:\n\n---\n{content}\n---"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    def create_completion(client, *args, **kwargs):
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1024,
        )
        return completion.choices[0].message['content']
    try:
        return call_llm_with_key_rotation(create_completion)
    except Exception as e:
        print(f"Error structuring data with LLM: {e}")
        # Create a fallback with simple links
        source_links = []
        for url in urls:
            try:
                # Attempt to get a nice name from the URL
                domain = urlparse(url).netloc.replace("www.", "")
                source_name = domain.split('.')[0]
            except:
                source_name = urlparse(url).netloc
            source_links.append(f"[{source_name}]({url})")
        
        sources_md = "\n".join(f"* {link}" for link in source_links)
        return f"I found some information, but had trouble summarizing it.\n\n**Sources:**\n{sources_md}"

@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/get_suggestions")
def get_suggestions():
    """
    Uses the LLM to generate a list of 4 creative suggestion prompts.
    """
    system_prompt = (
        "You are an AI assistant that generates creative and diverse suggestion prompts for a chat interface. "
        "Your goal is to provide users with interesting, knowledge-based questions to ask. The suggestions should cover a wide range of topics like science, history, technology, art, and creative tasks. "
        "IMPORTANT: Do NOT generate personal or conversational questions like 'What's your dream?' or 'How are you?'. Focus on prompts that help users discover information or test your creative abilities. "
        "For example: 'Explain the theory of relativity', 'What was the significance of the Silk Road?', or 'Write a short story about a city on Mars'. "
        "You must respond with only a valid JSON object and nothing else. The JSON object should be an array of 4 suggestion objects. "
        "Each object must have two keys: 'text' (the suggestion question, max 70 characters) and 'icon' (a valid Font Awesome 5 Free solid 'fas' icon name, like 'fa-brain' or 'fa-rocket')."
    )
    user_prompt = "Please generate 4 new, diverse, and knowledge-based suggestions."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    def create_completion(client, *args, **kwargs):
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=512,
            temperature=0.9
        )
        return completion.choices[0].message['content']
    try:
        response_content = call_llm_with_key_rotation(create_completion)
        # Clean potential markdown formatting
        if response_content.startswith("```json"):
            response_content = response_content[7:-4]
        suggestions = json.loads(response_content)
        return jsonify(suggestions)
    except Exception as e:
        print(f"Error generating suggestions with LLM: {e}")
        # Fallback to a default list of suggestions if the LLM fails
        default_suggestions = [
          { "text": "Explain the theory of relativity in simple terms", "icon": "fa-brain" },
          { "text": "What are the health benefits of meditation?", "icon": "fa-dna" },
          { "text": "Write a short story about a robot who discovers music", "icon": "fa-lightbulb" },
          { "text": "What was the significance of the Silk Road?", "icon": "fa-rocket" }
        ]
        return jsonify(default_suggestions)

@app.route("/chat", methods=["POST"])
def chat():
    global faiss_index
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"response": "Please enter a message."})

    # --- Casual Conversation Short-circuit ---
    casual_greetings = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'how are you?', 'what\'s up', 'whats up', 'yo', 'sup', 'greetings',
        'good night', 'goodbye', 'bye', 'see you', 'see ya', 'thanks', 'thank you'
    ]
    if user_message.strip().lower() in casual_greetings:
        import random
        friendly_responses = [
            "Hello! 👋 How can I help you today?",
            "Hi there! 😊 What would you like to know?",
            "Hey! Ready to explore some knowledge?",
            "Good day! Ask me anything.",
            "Hi! How can I assist you?"
        ]
        return jsonify({"response": random.choice(friendly_responses)})
    # --- End Casual Conversation Short-circuit ---

    try:
        # --- Bypass Cache for Current/Trending Queries ---
        current_keywords = [
            'current', 'latest', 'trending', 'right now', 'today', 'this week', 'this month', 'this year', 'breaking', 'recent', 'newest', 'news', 'live', 'upcoming', 'this season', 'this weekend', 'this evening', 'this morning', 'this afternoon'
        ]
        user_message_lower = user_message.lower()
        bypass_cache = any(kw in user_message_lower for kw in current_keywords)

        # --- Cache Check ---
        if not bypass_cache:
            print("Checking cache for similar query...")
            if len(vector_cache) > 0:
                query_emb = embedding_model.encode([user_message]).astype('float32')
                D, I = faiss_index.search(query_emb, 1)
                if D[0][0] < 0.2:  # L2 distance threshold for high similarity
                    _, doc_id, _, cached_response_html = vector_cache[I[0][0]]
                    if "Sources:" not in cached_response_html:
                        cached_response_html += "<p><i>(This response was retrieved from the cache)</i></p>"
                    return jsonify({"response": cached_response_html})
        # --- End of Cache Check ---

        # Step 1: Find relevant URLs
        urls = search_for_urls(user_message)
        
        # Handle search failure (e.g., timeout)
        if urls is None:
            return jsonify({"response": "Sorry, I'm having trouble connecting to the web to search for information. Please check your internet connection and try again."})

        # Handle case where no results were found
        if not urls:
            return jsonify({"response": f"Sorry, I couldn't find any relevant web pages for '{user_message}'."})

        # Step 2: Scrape all URLs and combine content
        all_content = []
        valid_urls = []
        # Define a character limit per URL to prevent the payload from being too large.
        CHAR_LIMIT_PER_URL = 8000
        for url in urls:
            scraped_data = scrape_and_return(url)
            content = scraped_data.get("content")
            if content and "Failed to retrieve content" not in content:
                # Truncate the content to the defined limit
                all_content.append(content[:CHAR_LIMIT_PER_URL])
                valid_urls.append(url)
        
        if not all_content:
            return jsonify({"response": f"Sorry, I was unable to retrieve content from the pages I found for '{user_message}'."})

        combined_content = "\n\n---\n\n".join(all_content)

        # Step 3: Structure the data with the LLM
        structured_summary_md = structure_data_with_llm(combined_content, user_message, valid_urls)
        
        # Step 4: Convert summary from Markdown to HTML for rendering
        bot_reply_html = md.render(structured_summary_md)

        # --- Cache Storage ---
        print("Storing new response in cache.")
        # We need a unique ID for each entry. A simple way is to use the user message hash.
        import hashlib
        doc_id = hashlib.md5(user_message.encode()).hexdigest()
        # Add to FAISS cache (optimized)
        new_emb = embedding_model.encode([user_message]).astype('float32')
        print("Embedding shape:", new_emb.shape)
        print("FAISS index dimension:", faiss_index.d)
        if new_emb.shape[1] != faiss_index.d:
            print("Dimension mismatch detected! Reinitializing FAISS index.")
            faiss_index = faiss.IndexFlatL2(new_emb.shape[1])
            EMBED_DIM = new_emb.shape[1]
        faiss_index.add(new_emb)
        vector_cache.append((new_emb[0], doc_id, user_message, bot_reply_html))
        if len(vector_cache) > MAX_CACHE_SIZE:
            # Remove the oldest vector from both FAISS and vector_cache
            remove_ids = np.array([0])
            faiss_index.remove_ids(remove_ids)
            vector_cache.pop(0)
        # --- End of Cache Storage ---

    except Exception as e:
        bot_reply_html = f"<p>An unexpected error occurred: {e}</p>"
        print("Error in /chat endpoint:")
        traceback.print_exc()
    
    return jsonify({"response": bot_reply_html})

@app.route("/test_pinecone_search")
def test_pinecone_search():
    query = "Famous historical structures and monuments"
    # Basic semantic search
    results = index.search(
        namespace="ns1",
        query={
            "top_k": 5,
            "inputs": {
                'text': query
            }
        }
    )
    if hasattr(results, "model_dump"):
        results = results.model_dump()
    elif hasattr(results, "to_dict"):
        results = results.to_dict()
    # Semantic search with reranking
    reranked_results = index.search(
        namespace="ns1",
        query={
            "top_k": 5,
            "inputs": {
                'text': query
            }
        },
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": 5,
            "rank_fields": ["chunk_text"]
        },
        fields=["category", "chunk_text"]
    )
    if hasattr(reranked_results, "model_dump"):
        reranked_results = reranked_results.model_dump()
    elif hasattr(reranked_results, "to_dict"):
        reranked_results = reranked_results.to_dict()
    return jsonify({
        "semantic_search": results,
        "reranked_search": reranked_results
    })

if __name__ == "__main__":
    # The reloader can cause issues with multiprocessing libraries like Selenium.
    # Disabling it makes the development server more stable for this use case.
    app.run(debug=True, use_reloader=False)