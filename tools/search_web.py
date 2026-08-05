import json
import os
import time

import requests

SERPER_KEY = os.environ.get("SERPER_API_KEY", "")

TOOL_SCHEMA = json.dumps({
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Array of query strings. Include multiple complementary search queries in a single call."
                }
            },
            "required": ["query"]
        }
    }
})

def google_search_with_serp(query: str):
    if not SERPER_KEY:
        return "SERPER_API_KEY is not set."

    def contains_chinese_basic(text: str) -> bool:
        return any('\u4E00' <= char <= '\u9FFF' for char in text)
    if contains_chinese_basic(query):
        payload = json.dumps({
            "q": query,
            "location": "China",
            "gl": "cn",
            "hl": "zh-cn"
        })

    else:
        payload = json.dumps({
            "q": query,
            "location": "United States",
            "gl": "us",
            "hl": "en"
        })
    headers = {
            'X-API-KEY': SERPER_KEY,
            'Content-Type': 'application/json'
        }


    for i in range(5):
        try:
            res = requests.post(
                "https://google.serper.dev/search",
                data=payload,
                headers=headers,
                timeout=30,
            )
            break
        except Exception as e:
            print(e)
            if i == 4:
                return f"Google search Timeout, return None, Please try again later."
            time.sleep(0.5 * (i + 1))
            continue

    data = res.content

    # Check HTTP status code - raise exception if not 200
    if res.status_code != 200:
        error_msg = f"API request failed with status {res.status_code}"
        try:
            error_data = json.loads(data.decode("utf-8"))
            if "message" in error_data:
                error_msg += f": {error_data['message']}"
            elif "error" in error_data:
                error_msg += f": {error_data['error']}"
        except:
            error_msg += f": {data.decode('utf-8', errors='ignore')[:200]}"
        raise Exception(f"Search API error for query '{query}': {error_msg}. Please check API key and configuration.")
    
    try:
        results = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse API response for query '{query}': {str(e)}. Response: {data.decode('utf-8', errors='ignore')[:200]}")
    
    # Check for API error messages in response
    if "error" in results:
        error_msg = results.get("error", "Unknown error")
        raise Exception(f"Search API error for query '{query}': {error_msg}. Please check API key and configuration.")
    
    if "message" in results and "organic" not in results:
        # API returned an error message
        error_msg = results.get("message", "Unknown error")
        raise Exception(f"Search API error for query '{query}': {error_msg}. Please check API key and configuration.")

    try:
        if "organic" not in results:
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

        web_snippets = list()
        idx = 0
        if "organic" in results:
            for page in results["organic"]:
                idx += 1
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

        content = f"### A Google search for '{query}' found {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)
        return content
    except Exception as e:
        # Re-raise API errors (they should have been caught earlier, but just in case)
        if "API error" in str(e) or "API request failed" in str(e) or "Failed to parse API response" in str(e):
            raise
        # Check if it's the exception we raised for no results
        if "No results found" in str(e):
            return str(e)
        # Otherwise, it's an unexpected error
        print(f"Unexpected error processing search results for query '{query}': {e}")
        raise Exception(f"Error processing search results for '{query}': {str(e)}. Please try again.")



def search_with_serp(query: str):
    result = google_search_with_serp(query)
    return result

def search_web(query):
    if query is None:
        return "[Search] Invalid request format: missing 'query' field"

    if isinstance(query, str):
        # 单个查询
        response = search_with_serp(query)
    else:
        # 多个查询
        responses = []
        for q in query:
            responses.append(search_with_serp(q))
        response = "\n\n---\n\n".join(responses)
    
    return response
