import json
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

from api_call import generate_response
from paper_formalization import (
    formalize_file,
    extract_json_from_response,
    save_to_json
)

SEMANTIC_SCHOLAR_API_KEY = "c7Xd3p89Ta2T7XFGTzPSi7u3kDRMsHAI8RwlXlFe"
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
UPLOADS_DIR = "../uploads"


def safe_get(url: str, **kwargs) -> dict:
    headers = kwargs.pop('headers', {}) or {}
    headers['x-api-key'] = SEMANTIC_SCHOLAR_API_KEY
    resp = requests.get(url, headers=headers, **kwargs)
    resp.raise_for_status()
    return resp.json()


def search_paper_by_title(title: str) -> dict:
    params = {'query': title, 'fields': 'paperId,openAccessPdf', 'limit': 1}
    time.sleep(1.1) # max 1 req/sec on s2 api key
    data = safe_get(f"{BASE_URL}/search", params=params)
    items = data.get('data', [])
    return items[0] if items else {}


def get_pdf_url_for_title(title: str) -> str | None:
    entry = search_paper_by_title(title)
    pdf_info = entry.get('openAccessPdf') or {}
    return pdf_info.get('url')


def find_seminal_papers(axioms: list[str]) -> list[dict]:
    prompt = (
        "You are a research assistant tasked with identifying foundational works.\n"
        "Given the following list of axioms, return a JSON array where each object maps the axiom\n"
        "to the most relevant seminal paper that first introduced or popularized the idea.\n\n"
        "Each object should contain:\n"
        "  - \"axiom\": the original axiom\n"
        "  - \"paper_title\": the title of the seminal paper\n\n"
        "Return ONLY a valid JSON array. No explanations, no markdown, no preamble.\n\n"
        "Axioms:\n"
        + "\n".join(f"- {ax}" for ax in axioms)
    )
    msg = generate_response(prompt)
    content = getattr(msg, 'content', str(msg))
    parsed = extract_json_from_response(content)

    if isinstance(parsed, list):
        return parsed

    # nice to have to verify responses
    print("[warn] JSON parsing failed. Falling back to line-based extraction.")
    print("[raw LLM content]:", content[:300]) 

    results = []
    for line in content.splitlines():
        if '->' in line:
            ax, title = map(str.strip, line.split('->', 1))
            results.append({"axiom": ax, "paper_title": title})
    return results



def process_paper(path_or_url: str) -> dict:
    """Formalize a file at a URL or local path and discover seminal papers."""
    result = {'url': path_or_url, 'axioms': [], 'seminal': []}
    data = formalize_file(path_or_url)
    axioms = data.get('axioms', [])
    result['axioms'] = axioms
    seminal = find_seminal_papers(axioms)
    for item in seminal:
        title = item.get('paper_title')
        pdf_url = get_pdf_url_for_title(title)
        if pdf_url:
            result['seminal'].append({'axiom': item.get('axiom'), 'title': title, 'pdf_url': pdf_url})
    return result


def recursive_process(start_path_or_url: str, max_depth: int = 1) -> list[dict]:
    results = []
    to_process = [start_path_or_url]
    current_depth = 0

    while to_process and current_depth <= max_depth:
        print(f"--- Depth {current_depth} ---")
        next_inputs = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_paper, path): path for path in to_process}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    res = future.result()
                    print(f"Processed: {path}")
                    results.append(res)
                    for item in res.get('seminal', []):
                        pdf_url = item.get('pdf_url')
                        if pdf_url:
                            next_inputs.append(pdf_url)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
        to_process = next_inputs
        current_depth += 1

    return results


def main():
    user_input = input("Enter initial file URL or local filename in 'uploads/': ")
    path_or_url = user_input.strip()

    # absolute path if it's a local file
    if not path_or_url.startswith("http"):
        local_path = os.path.join(UPLOADS_DIR, path_or_url)
        if not os.path.exists(local_path):
            print(f"File not found: {local_path}")
            return
        path_or_url = local_path

    depth_str = input("Enter recursion depth (0 for only initial): ")
    try:
        max_depth = int(depth_str)
    except ValueError:
        max_depth = 1
    print(f"Running recursion up to depth {max_depth}")

    all_results = recursive_process(path_or_url, max_depth)
    save_to_json(all_results, filename='recursive_axioms_results.json')
    print("Done. Results saved to recursive_axioms_results.json")

if __name__ == '__main__':
    main()
