import os
import time
import random
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

# Load environment variables from .env file
load_dotenv()

# Define default values for constants used in function signatures
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1
DEFAULT_REQUEST_TIMEOUT = 30


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.

    Returns:
        bool: True if all required variables are set, False otherwise
    """
    required_vars = ["COHERE_API_KEY", "QDRANT_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False

    return True


def get_env_var(var_name: str, default_value: Optional[str] = None) -> Optional[str]:
    """
    Get an environment variable with an optional default value.

    Args:
        var_name (str): Name of the environment variable
        default_value (Optional[str]): Default value if the variable is not set

    Returns:
        Optional[str]: Value of the environment variable or default value
    """
    return os.getenv(var_name, default_value)


def get_env_var_as_int(var_name: str, default_value: int) -> int:
    """
    Get an environment variable and convert it to an integer.

    Args:
        var_name (str): Name of the environment variable
        default_value (int): Default value if the variable is not set or invalid

    Returns:
        int: Integer value of the environment variable or default value
    """
    try:
        value = os.getenv(var_name)
        return int(value) if value else default_value
    except ValueError:
        logger.warning(f"Invalid integer value for {var_name}, using default: {default_value}")
        return default_value


def get_env_var_as_bool(var_name: str, default_value: bool) -> bool:
    """
    Get an environment variable and convert it to a boolean.

    Args:
        var_name (str): Name of the environment variable
        default_value (bool): Default value if the variable is not set or invalid

    Returns:
        bool: Boolean value of the environment variable or default value
    """
    value = os.getenv(var_name, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    else:
        return default_value


def create_session_with_retries(
    retries: int = None,
    backoff_factor: float = 0.3,
    status_forcelist: Tuple[int, ...] = (500, 502, 504)
) -> requests.Session:
    """
    Create a requests session with retry configuration.

    Args:
        retries (int): Number of retries
        backoff_factor (float): Backoff factor for retries
        status_forcelist (Tuple[int, ...]): HTTP status codes to retry on

    Returns:
        requests.Session: Configured session with retry adapter
    """
    # Use the configured value if not provided
    if retries is None:
        retries = MAX_RETRIES  # This will be available when the function is called

    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        # Don't raise exception on failed requests, let caller handle
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def retry_with_backoff(
    func: Callable,
    max_retries: int = None,
    base_delay: float = None,
    max_delay: float = 60,
    exceptions: Tuple = (Exception,)
) -> Any:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func (Callable): Function to execute
        max_retries (int): Maximum number of retry attempts
        base_delay (float): Base delay in seconds
        max_delay (float): Maximum delay in seconds
        exceptions (Tuple): Exception types to catch and retry on

    Returns:
        Any: Result of the function call
    """
    # Use the configured values if not provided
    if max_retries is None:
        max_retries = MAX_RETRIES
    if base_delay is None:
        base_delay = RETRY_DELAY

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"Function failed after {max_retries} retries: {str(e)}")
                raise e

            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)  # Add up to 10% jitter
            total_delay = delay + jitter

            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {total_delay:.2f}s...")
            time.sleep(total_delay)

    # This line should never be reached due to the return in the loop
    # or the exception being raised, but included for completeness
    raise last_exception


def handle_api_error(response: requests.Response, context: str = "") -> None:
    """
    Handle API errors by logging appropriate messages.

    Args:
        response (requests.Response): The response object from the API call
        context (str): Context for the error message
    """
    if response.status_code >= 400:
        error_msg = f"API Error in {context}: {response.status_code} - {response.text}"
        if response.status_code == 429:
            logger.warning(error_msg)
        elif response.status_code >= 500:
            logger.error(error_msg)
        else:
            logger.warning(error_msg)


def safe_request(
    method: str,
    url: str,
    **kwargs
) -> Optional[requests.Response]:
    """
    Make a safe HTTP request with error handling.

    Args:
        method (str): HTTP method (GET, POST, etc.)
        url (str): URL to request
        **kwargs: Additional arguments to pass to requests
    Returns:
        Optional[requests.Response]: Response object if successful, None otherwise
    """
    try:
        # Use the session with retries
        session = create_session_with_retries()

        # Set default timeout if not provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = REQUEST_TIMEOUT

        # Add default headers if not provided
        headers = kwargs.get('headers', {})
        headers.update(HTTP_HEADERS)
        kwargs['headers'] = headers

        response = session.request(method, url, **kwargs)

        if response.status_code >= 400:
            handle_api_error(response, f"safe_request to {url}")
            return None

        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during request to {url}: {str(e)}")
        return None


def is_valid_url(url: str, base_domain: str = None) -> bool:
    """
    Check if a URL is valid and optionally belongs to the base domain.

    Args:
        url (str): URL to validate
        base_domain (str, optional): Base domain that URLs should belong to

    Returns:
        bool: True if URL is valid, and belongs to base domain if specified, False otherwise
    """
    try:
        parsed_url = urlparse(url)

        # Check if scheme is valid
        if parsed_url.scheme not in ['http', 'https']:
            logger.debug(f"Invalid scheme for URL: {url}")
            return False

        # Check if URL has a netloc (domain)
        if not parsed_url.netloc:
            logger.debug(f"Missing domain for URL: {url}")
            return False

        # If base domain is specified, check if URL belongs to it
        if base_domain and base_domain not in parsed_url.netloc:
            logger.debug(f"URL {url} does not belong to base domain {base_domain}")
            return False

        return True
    except Exception as e:
        logger.debug(f"Error validating URL {url}: {str(e)}")
        return False


def validate_and_sanitize_url(url: str) -> Optional[str]:
    """
    Validate and sanitize a URL, returning a clean version or None if invalid.

    Args:
        url (str): URL to validate and sanitize

    Returns:
        Optional[str]: Sanitized URL if valid, None otherwise
    """
    try:
        # Handle relative URLs by making them absolute
        if url.startswith('/'):
            url = urljoin(BOOK_BASE_URL, url)
        elif not url.startswith(('http://', 'https://')):
            # Assume relative URL if it doesn't start with a scheme
            url = urljoin(BOOK_BASE_URL, url)

        # Parse and rebuild to ensure proper formatting
        parsed = urlparse(url)

        # Validate the URL
        if not is_valid_url(url, urlparse(BOOK_BASE_URL).netloc):
            logger.warning(f"Invalid URL after sanitization: {url}")
            return None

        # Reconstruct the URL to ensure it's properly formatted
        sanitized_url = parsed.geturl()

        return sanitized_url
    except Exception as e:
        logger.error(f"Error sanitizing URL {url}: {str(e)}")
        return None


def get_all_urls() -> List[str]:
    """
    Collect all URLs from the book website by parsing the sitemap and crawling pages.

    Returns:
        List[str]: List of all valid URLs found on the book website
    """
    all_urls = set()
    base_domain = urlparse(BOOK_BASE_URL).netloc

    # First, try to get URLs from sitemap
    sitemap_urls = get_urls_from_sitemap(SITEMAP_URL, base_domain)
    for url in sitemap_urls:
        sanitized_url = validate_and_sanitize_url(url)
        if sanitized_url:
            all_urls.add(sanitized_url)
    logger.info(f"Found {len(sitemap_urls)} URLs from sitemap, {len(all_urls)} after validation")

    # Additional crawling logic could go here if needed
    # For now, we'll rely primarily on the sitemap

    # Convert to sorted list for consistent processing
    url_list = sorted(list(all_urls))
    logger.info(f"Total unique URLs collected: {len(url_list)}")

    return url_list


def get_urls_from_sitemap(sitemap_url: str, base_domain: str) -> List[str]:
    """
    Extract URLs from a sitemap.xml file.

    Args:
        sitemap_url (str): URL of the sitemap.xml file
        base_domain (str): Base domain that URLs should belong to

    Returns:
        List[str]: List of valid URLs extracted from the sitemap
    """
    urls = []

    try:
        logger.info(f"Fetching sitemap: {sitemap_url}")
        response = safe_request("GET", sitemap_url)
        if not response:
            logger.error(f"Failed to fetch sitemap: {sitemap_url}")
            return urls

        soup = BeautifulSoup(response.content, 'xml')

        # Look for <url><loc> elements in sitemap
        url_elements = soup.find_all('loc')
        for url_elem in url_elements:
            url = url_elem.get_text().strip()
            if url and is_valid_url(url, base_domain):
                urls.append(url)
            elif url:
                logger.debug(f"Invalid URL from sitemap: {url}")

        # Also check for sitemap references (nested sitemaps)
        sitemap_elements = soup.find_all('sitemap')
        for sitemap_elem in sitemap_elements:
            loc_elem = sitemap_elem.find('loc')
            if loc_elem:
                nested_sitemap_url = loc_elem.get_text().strip()
                if nested_sitemap_url and is_valid_url(nested_sitemap_url, base_domain):
                    logger.info(f"Found nested sitemap: {nested_sitemap_url}")
                    nested_urls = get_urls_from_sitemap(nested_sitemap_url, base_domain)
                    urls.extend(nested_urls)
                elif nested_sitemap_url:
                    logger.debug(f"Invalid nested sitemap URL: {nested_sitemap_url}")

    except Exception as e:
        logger.error(f"Error parsing sitemap {sitemap_url}: {str(e)}")

    logger.info(f"Extracted {len(urls)} URLs from sitemap: {sitemap_url}")
    return urls


def crawl_site_for_urls(start_url: str, max_pages: int = 100) -> List[str]:
    """
    Crawl the site to find all internal links (alternative method if sitemap is incomplete).

    Args:
        start_url (str): Starting URL for crawling
        max_pages (int): Maximum number of pages to crawl

    Returns:
        List[str]: List of URLs found by crawling
    """
    urls = set()
    to_visit = [start_url]
    visited = set()
    base_domain = urlparse(start_url).netloc

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)

        if current_url in visited:
            continue

        visited.add(current_url)
        urls.add(current_url)

        logger.info(f"Crawling: {current_url} ({len(visited)}/{max_pages})")

        response = safe_request("GET", current_url)
        if not response:
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all links on the page
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(current_url, href)

            # Validate and sanitize the URL
            validated_url = validate_and_sanitize_url(absolute_url)
            if validated_url and is_valid_url(validated_url, base_domain) and validated_url not in visited:
                if validated_url not in to_visit:
                    to_visit.append(validated_url)

    return list(urls)



# Constants and Configuration Variables
BOOK_BASE_URL = get_env_var("BOOK_BASE_URL", "https://arifabbas11.github.io/humanoid-robotics-book/")
SITEMAP_URL = get_env_var("SITEMAP_URL", "https://arifabbas11.github.io/humanoid-robotics-book/sitemap.xml")
COHERE_API_KEY = get_env_var("COHERE_API_KEY")
QDRANT_URL = get_env_var("QDRANT_URL")
QDRANT_API_KEY = get_env_var("QDRANT_API_KEY")

# Processing configuration
CHUNK_SIZE = get_env_var_as_int("CHUNK_SIZE", 512)
CHUNK_OVERLAP = get_env_var_as_int("CHUNK_OVERLAP", 50)

# Qdrant configuration
QDRANT_COLLECTION_NAME = "rag_embedding"
VECTOR_SIZE = 1024  # Cohere embedding dimension

# Request configuration
REQUEST_TIMEOUT = get_env_var_as_int("REQUEST_TIMEOUT", 30)
MAX_RETRIES = get_env_var_as_int("MAX_RETRIES", 3)
RETRY_DELAY = get_env_var_as_int("RETRY_DELAY", 1)  # seconds

# Logging configuration
LOG_LEVEL = get_env_var("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_docusaurus_content(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """
    Extract content from Docusaurus-specific elements.

    Args:
        soup (BeautifulSoup): BeautifulSoup object containing the page content

    Returns:
        Optional[BeautifulSoup]: BeautifulSoup object with Docusaurus-specific content, or None if not found
    """
    # Try to find the main content area using Docusaurus-specific selectors
    content_selectors = [
        "article.markdown",           # Docusaurus markdown articles
        "div.theme-doc-markdown",     # Docusaurus markdown content container
        "main div.container",         # Main content container
        ".markdown",                 # General markdown class
        ".container",                # General container
        "article"                    # General article tag
    ]

    for selector in content_selectors:
        content_element = soup.select_one(selector)
        if content_element:
            # Create a new soup object with just the content element
            new_soup = BeautifulSoup("", 'html.parser')
            new_soup.append(content_element)
            return new_soup

    # If no specific content area found, try to extract from main content areas
    main_elements = soup.find_all(['main', 'div', 'section'])
    for element in main_elements:
        if element.get('role') == 'main' or 'main' in element.get('class', []) or 'content' in element.get('class', []):
            new_soup = BeautifulSoup("", 'html.parser')
            new_soup.append(element)
            return new_soup

    return None


def is_docusaurus_page(soup: BeautifulSoup) -> bool:
    """
    Determine if the page is a Docusaurus-generated page.

    Args:
        soup (BeautifulSoup): BeautifulSoup object containing the page content

    Returns:
        bool: True if the page appears to be generated by Docusaurus, False otherwise
    """
    # Check for Docusaurus-specific elements, classes, or attributes
    docusaurus_indicators = [
        soup.find('div', class_='navbar'),
        soup.find('div', class_='sidebar'),
        soup.find('div', class_='theme-doc-markdown'),
        soup.find('div', class_='doc-page'),
        soup.find('div', class_='doc-item'),
        soup.select_one('nav[aria-label="Navigation bar"]'),
        soup.select_one('div[class*="doc"]'),
        soup.select_one('div[class*="theme"]')
    ]

    # Count how many Docusaurus indicators are present
    indicators_found = sum(1 for indicator in docusaurus_indicators if indicator is not None)

    # If at least 2 indicators are found, consider it a Docusaurus page
    return indicators_found >= 2


def extract_text_from_url(url: str) -> Optional[str]:
    """
    Extract clean text content from a URL while preserving semantic structure.

    Args:
        url (str): URL to extract content from

    Returns:
        Optional[str]: Clean text content extracted from the URL, or None if extraction fails
    """
    # Validate the URL before attempting to extract content
    if not is_valid_url(url):
        logger.error(f"Invalid URL provided for text extraction: {url}")
        return None

    try:
        logger.info(f"Extracting text from URL: {url}")

        response = safe_request("GET", url)
        if not response:
            logger.error(f"Failed to fetch content from URL: {url}")
            return None

        # Check if the response is HTML
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            logger.warning(f"URL does not appear to contain HTML content: {url} (Content-Type: {content_type})")
            return None

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Check if this is a Docusaurus page and extract content accordingly
        is_docusaurus = is_docusaurus_page(soup)
        if is_docusaurus:
            logger.debug(f"Docusaurus page detected for URL: {url}")
            content_soup = extract_docusaurus_content(soup)
            if content_soup:
                soup = content_soup

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside", "meta", "link"]):
            script.decompose()

        # Try to find the main content area using Docusaurus-specific selectors
        content_element = None
        for selector in DOCUSAURUS_CONTENT_SELECTORS:
            content_element = soup.select_one(selector)
            if content_element:
                break

        # If no specific content area found, use the body
        if not content_element:
            content_element = soup.find('body')

        if not content_element:
            logger.warning(f"No content found for URL: {url}")
            return None

        # Extract text while preserving some structure
        text_parts = []

        # Process different types of elements to preserve structure
        for element in content_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'div', 'span', 'code', 'pre', 'table', 'tr', 'td', 'th'], recursive=True):
            # Get the text content
            text = element.get_text(strip=True)

            if text:
                # Add appropriate spacing based on element type
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    text_parts.append(f"\n\n{element.name.upper()}: {text}\n")
                elif element.name == 'li':
                    text_parts.append(f"  - {text}\n")
                elif element.name in ['p', 'div']:
                    text_parts.append(f"{text}\n")
                elif element.name in ['code', 'pre']:
                    text_parts.append(f"```\n{text}\n```\n")
                elif element.name in ['th', 'td']:
                    # For table cells, add some formatting
                    text_parts.append(f"{text} | ")
                elif element.name == 'tr':
                    # Add new line after table rows
                    text_parts.append(f"\n")
                elif element.name == 'table':
                    # Add more space around tables
                    text_parts.append(f"\n\nTABLE: {text}\n\n")
                else:
                    text_parts.append(f"{text} ")

        # Join all text parts and clean up
        content = " ".join(text_parts)

        # Clean up multiple newlines and spaces
        import re
        content = re.sub(r'\n\s*\n', '\n\n', content)  # Replace multiple newlines with double newlines
        content = re.sub(r'[ \t]+', ' ', content)  # Replace multiple spaces/tabs with single space
        content = content.strip()

        # Additional validation: ensure we have meaningful content
        if len(content) < 10:  # Minimum content length check
            logger.warning(f"Content extracted from {url} appears to be too short ({len(content)} chars)")
            return None

        logger.info(f"Successfully extracted {len(content)} characters from {url}")
        return content

    except Exception as e:
        logger.error(f"Error extracting text from URL {url}: {str(e)}")
        return None


# Docusaurus-specific selectors for content extraction
DOCUSAURUS_CONTENT_SELECTORS = [
    "article.markdown",
    "div.theme-doc-markdown",
    "main div.container",
    ".markdown",
    ".container",
    "article"
]

# HTTP headers to mimic a real browser request
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks with specified size and overlap.

    Args:
        text (str): Text to chunk
        chunk_size (int): Size of each chunk in characters
        overlap (int): Number of overlapping characters between chunks

    Returns:
        List[str]: List of text chunks
    """
    if not text:
        logger.warning("Empty text provided for chunking")
        return []

    if chunk_size <= 0:
        logger.error(f"Invalid chunk_size: {chunk_size}. Must be positive.")
        return []

    if overlap < 0 or overlap >= chunk_size:
        logger.warning(f"Invalid overlap: {overlap}. Must be between 0 and chunk_size-1. Defaulting to 0.")
        overlap = 0

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        # Ensure we don't have incomplete chunks at the end
        if len(chunk) < chunk_size and start > 0:
            # If this is a small final chunk and we've already processed content,
            # include it only if it's substantial (more than half the chunk size)
            if len(chunk) < chunk_size // 2 and len(chunks) > 0:
                # Add the small remainder to the last chunk instead
                if len(chunks) > 0:
                    chunks[-1] = chunks[-1] + chunk
                break

        chunks.append(chunk)

        # Move start position by chunk_size minus overlap
        start = end - overlap

        # If we're near the end and the remaining text is small, we might need to adjust
        if text_len - end < overlap and overlap > 0:
            break

    logger.info(f"Text chunked into {len(chunks)} chunks of size ~{chunk_size} with overlap {overlap}")
    return chunks


def embed(texts: List[str]) -> Optional[List[List[float]]]:
    """
    Generate embeddings for a list of texts using Cohere.

    Args:
        texts (List[str]): List of texts to embed

    Returns:
        Optional[List[List[float]]]: List of embeddings (each embedding is a list of floats), or None if failed
    """
    if not texts:
        logger.warning("Empty list of texts provided for embedding")
        return None

    # Validate input texts
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            logger.error(f"Text at index {i} is not a string: {type(text)}")
            return None
        if len(text) == 0:
            logger.warning(f"Text at index {i} is empty, skipping...")
            texts[i] = " "  # Replace empty string with a space to avoid API issues

    # Check if COHERE_API_KEY is set
    if not COHERE_API_KEY:
        logger.error("COHERE_API_KEY not set in environment variables")
        return None

    # Cohere has limits on the number of texts per request
    MAX_BATCH_SIZE = 96  # Cohere's typical batch limit

    all_embeddings = []

    try:
        # Initialize Cohere client
        cohere_client = cohere.Client(COHERE_API_KEY)

        # Process texts in batches if needed
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i:i + MAX_BATCH_SIZE]

            logger.info(f"Generating embeddings for batch {i//MAX_BATCH_SIZE + 1}/{(len(texts)-1)//MAX_BATCH_SIZE + 1} ({len(batch)} texts)...")

            # Generate embeddings using Cohere with retry logic
            response = retry_with_backoff(
                lambda b=batch: cohere_client.embed(
                    texts=b,
                    model='embed-english-v3.0',  # Using Cohere's English embedding model
                    input_type='search_document'  # Specify that these are documents for search
                ),
                exceptions=(requests.RequestException, ConnectionError)
            )

            if response and hasattr(response, 'embeddings') and response.embeddings:
                batch_embeddings = response.embeddings
                all_embeddings.extend(batch_embeddings)
                logger.debug(f"Batch {i//MAX_BATCH_SIZE + 1}: Successfully generated {len(batch_embeddings)} embeddings")
            else:
                logger.error(f"Batch {i//MAX_BATCH_SIZE + 1}: Cohere API returned empty or invalid response")
                return None

        logger.info(f"Successfully generated embeddings: {len(all_embeddings)} vectors of dimension {len(all_embeddings[0]) if all_embeddings else 0}")
        return all_embeddings

    except Exception as e:
        logger.error(f"Cohere API error during embedding: {str(e)}")
        # Note: Cohere library doesn't have a specific CohereAPIError, so we catch general exceptions
        return None


def test_content_extraction_with_sample_urls(sample_urls: List[str] = None) -> bool:
    """
    Test content extraction with sample URLs.

    Args:
        sample_urls (List[str], optional): List of sample URLs to test. If None, uses default test URLs.

    Returns:
        bool: True if testing was successful, False otherwise
    """
    if sample_urls is None:
        # Use a few sample URLs from the book, or default test URLs
        urls = get_all_urls()
        if urls:
            sample_urls = urls[:3]  # Test with first 3 URLs from the site
        else:
            # Fallback test URLs - these would be from the actual book site
            sample_urls = [
                BOOK_BASE_URL,
                f"{BOOK_BASE_URL}intro",
                f"{BOOK_BASE_URL}ros-fundamentals/intro"
            ]

    print(f"\nTesting content extraction with {len(sample_urls)} sample URLs...")

    success_count = 0
    for i, url in enumerate(sample_urls, 1):
        print(f"\nTest {i}/{len(sample_urls)}: Extracting content from {url}")

        content = extract_text_from_url(url)
        if content:
            print(f"  ✓ Success: Extracted {len(content)} characters")
            # Print first 200 characters as a preview
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"  Preview: {preview}")

            # Test chunking
            chunks = chunk_text(content)
            print(f"  Chunked into {len(chunks)} pieces")

            # Test embedding generation if API key is available
            if COHERE_API_KEY:
                print(f"  Generating embeddings for {len(chunks)} chunks...")
                embeddings = embed(chunks[:5])  # Limit to first 5 chunks for testing
                if embeddings:
                    print(f"  ✓ Successfully generated {len(embeddings)} embeddings of dimension {len(embeddings[0]) if embeddings else 0}")
                else:
                    print(f"  ✗ Failed to generate embeddings")
            else:
                print(f"  → Skipping embedding test - COHERE_API_KEY not set")

            success_count += 1
        else:
            print(f"  ✗ Failed to extract content")

    print(f"\nContent extraction test results: {success_count}/{len(sample_urls)} successful")

    if success_count > 0:
        print("Content extraction is working properly.")
        return True
    else:
        print("Content extraction failed for all test URLs.")
        return False


def test_embedding_generation_with_sample_content() -> bool:
    """
    Test embedding generation with sample content.

    Returns:
        bool: True if embedding generation testing was successful, False otherwise
    """
    print(f"\nTesting embedding generation with sample content...")

    if not COHERE_API_KEY:
        print("  → Skipping embedding test - COHERE_API_KEY not set in environment")
        return False

    # Sample content for testing
    sample_texts = [
        "This is a test sentence for embedding generation.",
        "Artificial intelligence and machine learning are transforming technology.",
        "Robotics involves the design, construction, and operation of robots.",
        "Natural language processing enables computers to understand human language.",
        "Vector embeddings represent text in high-dimensional space for similarity matching."
    ]

    try:
        print(f"  Generating embeddings for {len(sample_texts)} sample texts...")
        embeddings = embed(sample_texts)

        if embeddings and len(embeddings) == len(sample_texts):
            print(f"  ✓ Successfully generated {len(embeddings)} embeddings")
            print(f"  Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")

            # Test similarity by comparing first two embeddings (should be different)
            if len(embeddings) >= 2:
                # Simple similarity check by comparing a few values
                first_emb = embeddings[0][:5]  # First 5 dimensions
                second_emb = embeddings[1][:5]  # First 5 dimensions
                print(f"  Sample embedding (first 5 dims) for text 1: {first_emb}")
                print(f"  Sample embedding (first 5 dims) for text 2: {second_emb}")

            return True
        else:
            print(f"  ✗ Failed to generate embeddings properly")
            return False

    except Exception as e:
        print(f"  ✗ Error during embedding test: {str(e)}")
        return False


def create_collection() -> bool:
    """
    Create a Qdrant collection for storing embeddings with specified parameters.

    Returns:
        bool: True if collection was created successfully (or already exists), False otherwise
    """
    if not QDRANT_URL:
        logger.error("QDRANT_URL not set in environment variables")
        return False

    if not QDRANT_API_KEY:
        logger.error("QDRANT_API_KEY not set in environment variables")
        return False

    try:
        logger.info(f"Connecting to Qdrant at {QDRANT_URL}...")

        # Initialize Qdrant client
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=REQUEST_TIMEOUT
        )

        # Check if collection already exists
        try:
            existing_collections = client.get_collections()
            collection_exists = any(col.name == QDRANT_COLLECTION_NAME for col in existing_collections.collections)
        except:
            # If we can't list collections, assume the collection doesn't exist
            collection_exists = False

        if collection_exists:
            logger.info(f"Collection '{QDRANT_COLLECTION_NAME}' already exists")
            return True

        # Create collection with specified parameters
        logger.info(f"Creating collection '{QDRANT_COLLECTION_NAME}' with vector size {VECTOR_SIZE}...")

        client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE  # Using cosine distance for semantic similarity
            ),
            # Set up HNSW indexing for efficient similarity search
            hnsw_config=models.HnswConfigDiff(
                m=16,  # Number of edges per vertex
                ef_construct=100,  # Construction time parameter
                full_scan_threshold=10000,  # Threshold for full scan vs index
            ),
            # Set up optimizers for efficient storage
            optimizers_config=models.OptimizersConfigDiff(
                deleted_threshold=0.2,  # Threshold for deleted vectors
                vacuum_min_vector_number=1000,  # Minimum number of vectors for vacuum
            )
        )

        logger.info(f"Successfully created collection '{QDRANT_COLLECTION_NAME}' with vector size {VECTOR_SIZE}")
        return True

    except Exception as e:
        logger.error(f"Error creating Qdrant collection: {str(e)}")
        return False


def save_chunk_to_qdrant(text_chunk: str, embedding: List[float], source_url: str = None, metadata: Dict = None) -> bool:
    """
    Save a text chunk with its embedding to Qdrant with idempotent operations.

    Args:
        text_chunk (str): The text chunk to store
        embedding (List[float]): The embedding vector for the text chunk
        source_url (str, optional): URL where the content was extracted from
        metadata (Dict, optional): Additional metadata to store with the chunk

    Returns:
        bool: True if the chunk was saved successfully (or already existed), False otherwise
    """
    if not QDRANT_URL:
        logger.error("QDRANT_URL not set in environment variables")
        return False

    if not QDRANT_API_KEY:
        logger.error("QDRANT_API_KEY not set in environment variables")
        return False

    if not text_chunk or not embedding:
        logger.error("Text chunk or embedding is empty")
        return False

    # Validate embedding dimension
    if len(embedding) != VECTOR_SIZE:
        logger.error(f"Embedding dimension mismatch: expected {VECTOR_SIZE}, got {len(embedding)}")
        return False

    # Generate a unique ID for this chunk based on its content
    import hashlib
    chunk_id = hashlib.md5(f"{text_chunk}{str(embedding)[:50]}".encode()).hexdigest()

    def _check_existing_chunk():
        """Check if a chunk with the same ID already exists in Qdrant."""
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=REQUEST_TIMEOUT
        )

        try:
            # Try to retrieve the point by ID
            records = client.retrieve(
                collection_name=QDRANT_COLLECTION_NAME,
                ids=[chunk_id]
            )
            return len(records) > 0
        except Exception:
            # If there's an error (e.g., collection doesn't exist), return False
            return False

    def _save_chunk_operation():
        """Internal function to perform the actual save operation."""
        # Initialize Qdrant client
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=REQUEST_TIMEOUT
        )

        # Prepare payload with the text chunk and any additional metadata
        payload = {
            "text": text_chunk,
            "source_url": source_url or "",
            "timestamp": time.time(),
            "chunk_length": len(text_chunk)
        }

        # Add any additional metadata
        if metadata:
            payload.update(metadata)

        # Upsert the point (this will update if it exists, or create if it doesn't)
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        return True

    try:
        logger.info(f"Checking for existing chunk in Qdrant: ID={chunk_id[:8]}...")

        # Check if the chunk already exists (idempotent operation)
        if _check_existing_chunk():
            logger.info(f"Chunk already exists in Qdrant (idempotent operation): ID={chunk_id[:8]}...")
            return True

        logger.info(f"Saving new chunk to Qdrant: ID={chunk_id[:8]}..., length={len(text_chunk)} chars, embedding size={len(embedding)}")

        # Use retry logic for the save operation
        result = retry_with_backoff(
            _save_chunk_operation,
            exceptions=(Exception,)  # Retry on any exception during the save operation
        )

        if result:
            logger.info(f"Successfully saved chunk to Qdrant: ID={chunk_id[:8]}...")
            return True
        else:
            logger.error(f"Failed to save chunk to Qdrant: ID={chunk_id[:8]}...")
            return False

    except Exception as e:
        logger.error(f"Error in idempotent save operation: {str(e)}")
        if "404" in str(e) or "not found" in str(e).lower():
            logger.error("Collection may not exist. Try calling create_collection() first.")
        return False


def process_single_url(url: str) -> bool:
    """
    Process a single URL through the entire pipeline: extract -> chunk -> embed -> store.

    Args:
        url (str): URL to process

    Returns:
        bool: True if the URL was processed successfully, False otherwise
    """
    logger.info(f"Processing URL: {url}")

    # Validate the URL first
    if not is_valid_url(url):
        logger.error(f"Invalid URL provided: {url}")
        return False

    try:
        # Step 1: Extract content from the URL
        logger.info(f"Step 1: Extracting content from {url}")
        content = extract_text_from_url(url)
        if not content:
            logger.error(f"Failed to extract content from {url}")
            return False

        logger.info(f"Extracted {len(content)} characters from {url}")

        # Step 2: Chunk the content
        logger.info(f"Step 2: Chunking content (size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})")
        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            logger.error(f"No chunks generated from content in {url}")
            return False

        logger.info(f"Content chunked into {len(chunks)} pieces")

        # Step 3: Generate embeddings for each chunk
        logger.info(f"Step 3: Generating embeddings for {len(chunks)} chunks")
        embeddings = embed(chunks)
        if not embeddings or len(embeddings) != len(chunks):
            logger.error(f"Failed to generate embeddings for {url} (expected {len(chunks)}, got {len(embeddings) if embeddings else 0})")
            return False

        logger.info(f"Generated {len(embeddings)} embeddings successfully")

        # Step 4: Store each chunk with its embedding in Qdrant
        logger.info(f"Step 4: Storing {len(chunks)} chunks in Qdrant")
        success_count = 0
        failed_chunks = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Add metadata for this chunk
            metadata = {
                "source_url": url,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "processing_timestamp": time.time()
            }

            if save_chunk_to_qdrant(chunk, embedding, source_url=url, metadata=metadata):
                success_count += 1
                logger.debug(f"Saved chunk {i+1}/{len(chunks)} for {url}")
            else:
                logger.warning(f"Failed to save chunk {i+1}/{len(chunks)} for {url}")
                failed_chunks.append(i)

        logger.info(f"Successfully stored {success_count}/{len(chunks)} chunks for {url}")

        # Log details about failed chunks if any
        if failed_chunks:
            logger.warning(f"Failed to save {len(failed_chunks)} chunks for {url} at indices: {failed_chunks}")

        if success_count == len(chunks):
            logger.info(f"Successfully processed all {len(chunks)} chunks for {url}")
            return True
        elif success_count > 0:
            logger.warning(f"Partially processed {url}: {success_count}/{len(chunks)} chunks saved")
            return True  # Consider partial success as success
        else:
            logger.error(f"Failed to save any chunks for {url}")
            return False

    except KeyboardInterrupt:
        logger.error("Processing interrupted by user")
        raise  # Re-raise to allow proper cleanup
    except Exception as e:
        logger.error(f"Unexpected error processing URL {url}: {str(e)}")
        return False


def process_all_urls(urls: List[str] = None, max_pages: int = None) -> Dict[str, Any]:
    """
    Process all URLs through the entire pipeline with progress tracking.

    Args:
        urls (List[str], optional): List of URLs to process. If None, gets all URLs from the site.
        max_pages (int, optional): Maximum number of pages to process. If None, processes all.

    Returns:
        Dict[str, Any]: Results dictionary with statistics and success/failure counts
    """
    logger.info("Starting to process all URLs through the pipeline")

    try:
        # Get URLs if not provided
        if urls is None:
            logger.info("Getting all URLs from the book website...")
            urls = get_all_urls()
            if not urls:
                logger.error("No URLs found to process")
                return {"success": False, "error": "No URLs found"}

        # Limit number of pages if specified
        if max_pages and len(urls) > max_pages:
            logger.info(f"Limiting to first {max_pages} URLs out of {len(urls)} total")
            urls = urls[:max_pages]

        logger.info(f"Processing {len(urls)} URLs through the pipeline...")

        # Ensure Qdrant collection exists before processing
        logger.info("Creating/verifying Qdrant collection...")
        if not create_collection():
            logger.error("Failed to create Qdrant collection")
            return {"success": False, "error": "Failed to create Qdrant collection"}

        # Process each URL with progress tracking
        results = {
            "total_urls": len(urls),
            "successful": 0,
            "failed": 0,
            "failed_urls": [],
            "start_time": time.time(),
            "completed_urls": []
        }

        for i, url in enumerate(urls, 1):
            try:
                # Print a visual progress indicator every 10 URLs or at the start
                if i == 1 or i % 10 == 0 or i == len(urls):
                    progress = (i / len(urls)) * 100
                    bar_length = 30
                    filled_length = int(bar_length * i // len(urls))
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    print(f"\rProgress: |{bar}| {i}/{len(urls)} URLs ({progress:.1f}%) - Success: {results['successful']}, Failed: {results['failed']}", end='', flush=True)

                logger.info(f"Processing URL {i}/{len(urls)}: {url}")

                # Process the URL
                success = process_single_url(url)

                if success:
                    results["successful"] += 1
                    results["completed_urls"].append(url)
                    logger.info(f"Successfully processed URL {i}/{len(urls)}")
                else:
                    results["failed"] += 1
                    results["failed_urls"].append(url)
                    logger.warning(f"Failed to process URL {i}/{len(urls)}")

            except KeyboardInterrupt:
                logger.error(f"Processing interrupted by user at URL {i}/{len(urls)}")
                print(f"\nProcessing interrupted by user.")
                break  # Break the loop but continue to return partial results
            except Exception as e:
                logger.error(f"Unexpected error processing URL {i}/{len(urls)}: {str(e)}")
                results["failed"] += 1
                results["failed_urls"].append(url)
                continue  # Continue with the next URL

        # Print final progress bar
        print()  # New line after progress bar
        logger.info(f"Progress: {results['successful']}/{len(urls)} URLs processed successfully")

        # Calculate and log final statistics
        results["end_time"] = time.time()
        results["total_time"] = results["end_time"] - results["start_time"]
        results["success_rate"] = (results["successful"] / len(urls)) * 100 if urls else 0
        results["avg_time_per_url"] = results["total_time"] / len(urls) if urls else 0
        results["success"] = results["successful"] > 0  # Consider overall success if at least one URL was processed

        logger.info(f"Pipeline completed. Processed {results['successful']}/{len(urls)} URLs successfully in {results['total_time']:.2f} seconds")
        logger.info(f"Average time per URL: {results['avg_time_per_url']:.2f} seconds")
        logger.info(f"Success rate: {results['success_rate']:.1f}%")

        # Print summary
        print(f"\n--- Pipeline Summary ---")
        print(f"Total URLs processed: {len(urls)}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Time elapsed: {results['total_time']:.2f} seconds")
        print(f"Average time per URL: {results['avg_time_per_url']:.2f} seconds")
        if results['failed'] > 0:
            print(f"Failed URLs: {results['failed_urls'][:5]}{'...' if len(results['failed_urls']) > 5 else ''}")  # Show first 5 failed URLs

        return results

    except KeyboardInterrupt:
        logger.error("Entire pipeline interrupted by user")
        print(f"\nEntire pipeline interrupted by user.")
        # Return partial results if available
        if 'results' in locals():
            results["end_time"] = time.time()
            results["total_time"] = results["end_time"] - results["start_time"]
            results["success_rate"] = (results["successful"] / len(urls)) * 100 if urls else 0
            results["avg_time_per_url"] = results["total_time"] / len(urls) if urls else 0
            return results
        else:
            return {"success": False, "error": "Interrupted before processing started"}
    except Exception as e:
        logger.error(f"Unexpected error in process_all_urls: {str(e)}")
        return {"success": False, "error": str(e)}


def test_vector_storage_with_sample_embeddings() -> bool:
    """
    Test vector storage with sample embeddings.

    Returns:
        bool: True if vector storage testing was successful, False otherwise
    """
    print(f"\nTesting vector storage with sample embeddings...")

    if not QDRANT_URL or not QDRANT_API_KEY:
        print("  → Skipping vector storage test - QDRANT credentials not set in environment")
        return False

    # First, ensure the collection exists
    print("  Creating/verifying Qdrant collection...")
    if not create_collection():
        print("  ✗ Failed to create Qdrant collection")
        return False

    # Sample content for testing
    sample_texts = [
        "The robot navigated through the hallway using its sensors.",
        "Machine learning algorithms help robots adapt to new environments.",
        "Artificial intelligence enables robots to make autonomous decisions.",
        "Computer vision allows robots to recognize objects and obstacles.",
        "Natural language processing enables human-robot interaction."
    ]

    try:
        print(f"  Generating embeddings for {len(sample_texts)} sample texts...")
        embeddings = embed(sample_texts)

        if not embeddings or len(embeddings) != len(sample_texts):
            print(f"  ✗ Failed to generate embeddings properly")
            return False

        print(f"  Successfully generated {len(embeddings)} embeddings")

        # Test saving each embedding to Qdrant
        success_count = 0
        for i, (text, embedding) in enumerate(zip(sample_texts, embeddings)):
            print(f"  Saving sample {i+1}/{len(sample_texts)} to Qdrant...")

            # Add metadata for testing
            metadata = {
                "test_chunk": True,
                "sample_id": i,
                "category": "test_data"
            }

            if save_chunk_to_qdrant(text, embedding, source_url="test://sample", metadata=metadata):
                print(f"    ✓ Saved chunk {i+1} successfully")
                success_count += 1
            else:
                print(f"    ✗ Failed to save chunk {i+1}")

        print(f"\nVector storage test results: {success_count}/{len(sample_texts)} successful")

        if success_count > 0:
            print("Vector storage is working properly.")
            return True
        else:
            print("Vector storage failed for all test samples.")
            return False

    except Exception as e:
        print(f"  ✗ Error during vector storage test: {str(e)}")
        return False


def main():
    print("Book RAG Content Ingestion Pipeline initialized")
    print(f"Base URL: {BOOK_BASE_URL}")
    print(f"Chunk size: {CHUNK_SIZE}, Chunk overlap: {CHUNK_OVERLAP}")

    # Validate environment
    if validate_environment():
        print("Environment validation passed")
    else:
        print("Environment validation failed - missing required variables")

    # Test get_all_urls function
    print("\nTesting get_all_urls function...")
    urls = get_all_urls()
    print(f"Found {len(urls)} URLs")
    if urls:
        print("First 5 URLs:")
        for url in urls[:5]:
            print(f"  - {url}")

    # Test content extraction with sample URLs
    test_content_extraction_with_sample_urls()

    # Test embedding generation with sample content
    test_embedding_generation_with_sample_content()

    # Test vector storage with sample embeddings
    test_vector_storage_with_sample_embeddings()


if __name__ == "__main__":
    main()
