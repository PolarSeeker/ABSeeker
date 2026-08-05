import json
import os
import random
import time
import asyncio
import requests
import tiktoken
import httpx
from openai import OpenAI

_INTERNAL_GATEWAY_HTTP_CLIENT = None


def _internal_gateway_http_client():
    global _INTERNAL_GATEWAY_HTTP_CLIENT
    if _INTERNAL_GATEWAY_HTTP_CLIENT is None:
        _INTERNAL_GATEWAY_HTTP_CLIENT = httpx.Client(trust_env=False, timeout=300.0)
    return _INTERNAL_GATEWAY_HTTP_CLIENT

SUMMARY_APIS = [
    ("http://127.0.0.1:8890/v1", "EMPTY")
]

VISIT_SERVER_TYPE = os.getenv("VISIT_SERVER", "local").lower()

DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://your-llm-gateway.example.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

GPU_GEEK_URL = os.getenv("GPUGEEK_URL", "https://api.gpugeek.com/v1")
GPU_GEEK_API_KEY = os.getenv("GPUGEEK_API_KEY", "")
GPU_GEEK_MODEL = os.getenv("GPUGEEK_MODEL", "Vendor3/DeepSeek-V4-Flash")

GPT_OSS_URL = os.getenv("GPT_OSS_URL", "http://127.0.0.1:3000/v1")
GPT_OSS_API_KEY = os.getenv("GPT_OSS_API_KEY", "")
GPT_OSS_MODEL = os.getenv("GPT_OSS_MODEL", "gpt-oss-20b")


def call_server(msgs, max_retries=2):
    # os.environ['http_proxy'] = 'http://127.0.0.1:7891'
    # os.environ['https_proxy'] = 'http://127.0.0.1:7891'

    _visit_server = os.getenv("VISIT_SERVER", "local").lower()
    if _visit_server == "ds":
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_URL,
            http_client=_internal_gateway_http_client(),
        )
        model_name = os.environ.get("SUMMARY_MODEL_NAME", DEEPSEEK_MODEL)
    elif _visit_server == "gpugeek":
        client = OpenAI(api_key=GPU_GEEK_API_KEY, base_url=GPU_GEEK_URL)
        model_name = os.environ.get("SUMMARY_MODEL_NAME", GPU_GEEK_MODEL)
    elif _visit_server == "oss":
        client = OpenAI(api_key=GPT_OSS_API_KEY, base_url=GPT_OSS_URL)
        model_name = os.environ.get("SUMMARY_MODEL_NAME", GPT_OSS_MODEL)
    else:  # local
        url_llm, api_key = random.choice(SUMMARY_APIS)
        model_name = os.environ.get("SUMMARY_MODEL_NAME", "gpt-oss-120b")
        client = OpenAI(api_key=api_key, base_url=url_llm)

    for attempt in range(max_retries):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=msgs,
                temperature=0.7
            )
            content = chat_response.choices[0].message.content
            if not content:
                content = getattr(chat_response.choices[0].message, 'reasoning_content', None) or ""
            if content:
                try:
                    json.loads(content)
                except Exception:
                    left = content.find('{')
                    right = content.rfind('}')
                    if left != -1 and right != -1 and left <= right:
                        content = content[left:right+1]
                return content
        except Exception:
            if attempt == (max_retries - 1):
                return ""
            continue

VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))

JINA_API_KEYS = os.getenv(
    "JINA_API_KEY"
)
HTTP_PROXY = os.getenv("VISIT_HTTP_PROXY", "http://127.0.0.1:6478")
VISIT_REQUESTS_PROXIES = {"http": HTTP_PROXY, "https": HTTP_PROXY}

TOOL_SCHEMA = json.dumps({
    "type": "function",
    "function": {
        "name": "visit_web",
        "description": "Visit webpage(s) and return the summary of the content.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
                },
                "goal": {
                    "type": "string",
                    "description": "The goal of the visit for webpage(s)."
                }
            },
            "required": ["url", "goal"]
        }
    }
})

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def jina_readpage(url: str) -> str:
    """
    Read webpage content using Jina service.

    Args:
        url: The URL to read
        goal: The goal/purpose of reading the page

    Returns:
        str: The webpage content or error message
    """
    max_retries = 3
    timeout = 50

    for attempt in range(max_retries):
        headers = {
            "Authorization": f"Bearer {JINA_API_KEYS}",
        }
        try:
            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=timeout,
                proxies=VISIT_REQUESTS_PROXIES,
            )
            if response.status_code == 200:
                webpage_content = response.text
                return webpage_content
            else:
                raise ValueError("jina readpage error")
        except Exception as e:
            time.sleep(0.5)
            if attempt == max_retries - 1:
                return "[visit] Failed to read page."

    return "[visit] Failed to read page."


def html_readpage_jina(url: str) -> str:
    max_attempts = 8
    for attempt in range(max_attempts):
        content = jina_readpage(url)
        if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            return content
    return "[visit] Failed to read page."


def readpage_jina(url: str, goal: str) -> str:
    """
    Attempt to read webpage content by alternating between jina and aidata services.

    Args:
        url: The URL to read
        goal: The goal/purpose of reading the page

    Returns:
        str: The webpage content or error message
    """
    max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 3))

    content = html_readpage_jina(url)

    if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
        content = truncate_to_tokens(content, max_tokens=95000)
        messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
        parse_retry_times = 0
        raw = call_server(messages, max_retries=max_retries)
        summary_retries = 3
        while len(raw) < 10 and summary_retries >= 0:
            truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
            status_msg = (
                f"[visit] Summary url[{url}] "
                f"attempt {3 - summary_retries + 1}/3, "
                f"content length: {len(content)}, "
                f"truncating to {truncate_length} chars"
            ) if summary_retries > 0 else (
                f"[visit] Summary url[{url}] failed after 3 attempts, "
                f"final truncation to 25000 chars"
            )
            print(status_msg)
            content = content[:truncate_length]
            extraction_prompt = EXTRACTOR_PROMPT.format(
                webpage_content=content,
                goal=goal
            )
            messages = [{"role": "user", "content": extraction_prompt}]
            raw = call_server(messages, max_retries=max_retries)
            summary_retries -= 1

        parse_retry_times = 2
        if isinstance(raw, str):
            raw = raw.replace("```json", "").replace("```", "").strip()
        while parse_retry_times < 3:
            try:
                raw = json.loads(raw)
                break
            except:
                raw = call_server(messages, max_retries=max_retries)
                parse_retry_times += 1

        if parse_retry_times >= 3:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                url=url, goal=goal
            )
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                url=url, goal=goal
            )
            useful_information += "Evidence in page: \n" + str(raw.get("evidence", "")) + "\n\n"
            useful_information += "Summary: \n" + str(raw.get("summary", "")) + "\n\n"

        if len(useful_information) < 10 and summary_retries < 0:
            print("[visit] Could not generate valid summary after maximum retries")
            useful_information = "[visit] Failed to read page"

        return useful_information
    else:
        return f"[visit] Failed to visit the url: {url}"


async def visit_web(url, goal):
    if url is None or goal is None:
        return "[Visit] Invalid request format: missing 'url' or 'goal' field"

    start_time = time.time()

    # Create log folder if it doesn't exist
    log_folder = "log"
    os.makedirs(log_folder, exist_ok=True)

    if isinstance(url, str):
        response = await asyncio.to_thread(readpage_jina, url, goal)
    else:
        response = []
        start_time = time.time()
        for u in url:
            if time.time() - start_time > 900:
                cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                    url=url, goal=goal
                )
                cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            else:
                try:
                    cur_response = await asyncio.to_thread(readpage_jina, u, goal)
                except Exception as e:
                    cur_response = f"Error fetching {u}: {str(e)}"
            response.append(cur_response)
        response = "\n---\n".join(response)

    return response.strip()
