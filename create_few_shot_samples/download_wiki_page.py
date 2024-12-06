import requests
from bs4 import BeautifulSoup

def get_wikipedia_content_as_json(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch the Wikipedia page. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    content = []
    
    content_div = soup.find('div', {'class': 'mw-parser-output'})
    if not content_div:
        raise ValueError("Could not find the main content of the Wikipedia page.")
    
    for element in content_div.find_all(['p', 'img', 'figure']):
        if element.name == 'p':  # Text content
            text = element.get_text(strip=True)
            if text:
                content.append({"type": "text", "value": text})
        elif element.name == 'img':  # Image content
            img_src = element.get('src')
            if img_src and not img_src.startswith('http'):
                img_src = 'https:' + img_src
            caption = None
            parent_figure = element.find_parent('figure')
            if parent_figure:
                caption_elem = parent_figure.find('figcaption')
                if caption_elem:
                    caption = caption_elem.get_text(strip=True)
            content.append({"type": "image", "value": img_src})
            if caption:
                content.append({"type": "text", "value": caption})
    
    return content
