from pdb import runcall
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import pypandoc

HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
}

def clean_url(url):
    """
    截取 # 之前的部分，去除 fragment
    """
    return url.split('#')[0]

def is_valid_page(url):
    # 只下载路径中包含 _books_ml 的网页
    parsed = urlparse(url)
    return "ml-q" in parsed.path

def save_image(img_url, save_dir="docs/images"):
    try:
        os.makedirs(save_dir, exist_ok=True)
        filename = urlparse(img_url).path.split('/')[-1]
        save_path = os.path.join(save_dir, filename)
        if not os.path.exists(save_path):
            resp = requests.get(img_url, headers=HEADERS)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(resp.content)
            print(f"Image saved: {save_path}")
        return save_path
    except Exception as e:
        print(f"Error saving image {img_url}: {e}")

def html_to_markdown(html, page_dir="pages", img_dir="images"):
    """
    Convert HTML to Markdown using pypandoc.
    """
    # pypandoc will keep local image and code block formatting intact
    md = pypandoc.convert_text(html, 'md', format='html')
    return md

def fetch_and_save(url, visited, queue, save_dir="docs", img_dir="docs/images"):
    print(f"Fetching: {url}")
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        html = response.text
        # 保存页面
        path = urlparse(url).path.replace('/', '_') or 'index'
        html_filename = f"{save_dir}/{path}.html"
        md_filename = f"{save_dir}/{path}.md"
        os.makedirs(save_dir, exist_ok=True)
        # 解析页面内容
        soup = BeautifulSoup(html, "html.parser")
        # 下载并替换图片链接
        for img in soup.find_all("img", src=True):
            img_url = urljoin(url, img["src"])
            local_img_path = save_image(img_url, save_dir=img_dir)
            if local_img_path:
                img['src'] = os.path.relpath(local_img_path, start=save_dir)
        # 保存修改后的 HTML
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(str(soup))
        print(f"Page saved: {html_filename}")
        # 转换并保存为 Markdown
        md_content = html_to_markdown(str(soup), page_dir=save_dir, img_dir=img_dir)
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Markdown saved: {md_filename}")
        # 解析链接，加入待抓取队列
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a['href'])
            if is_valid_page(link) and link not in visited and link.startswith("http"):
                queue.append(link)
    except Exception as e:
        print(f"Error fetching {url}: {e}")

def main(start_url):
    visited = set()
    queue = [start_url]
    while queue:
        url = queue.pop(0)
        url = clean_url(url)
        if url in visited:
            continue
        visited.add(url)
        if is_valid_page(url):
            fetch_and_save(url, visited, queue)
        else: queue.append(url) 

    print("Done.")

if __name__ == "__main__":
    # 替换成你要抓取的起始页面
    main("https://sebastianraschka.com/books/ml-q-and-ai/")