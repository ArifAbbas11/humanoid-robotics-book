from main import process_all_urls, get_all_urls

# Get a few sample URLs
urls = get_all_urls()[:5]  # Just first 5 URLs for quick test
print(f'Processing {len(urls)} URLs...')

# Process the URLs
results = process_all_urls(urls=urls, max_pages=2)  # Limit to first 2 for quick test
print('Processing complete!')
print(f'Successful: {results["successful"]}, Failed: {results["failed"]}')

