import asyncio
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

async def fetch_html(url, page):
    await page.goto(url)
    await page.wait_for_load_state('networkidle')
    html_content = await page.content()
    return html_content

def parse_html_to_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator='\n')
    return text

async def save_documentation(url, output_folder):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Ensure the output folder exists
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Fetch the main page
        main_html = await fetch_html(url, page)
        main_text = parse_html_to_text(main_html)

        # Save the main page content
        main_file_path = output_path / "index.txt"
        with open(main_file_path, "w", encoding="utf-8") as file:
            file.write(main_text)

        # Find all links to other documentation pages
        soup = BeautifulSoup(main_html, 'html.parser')
        links = soup.find_all('a', href=True)
        doc_links = set(link['href'] for link in links if link['href'].startswith(url))

        for link in doc_links:
            html_content = await fetch_html(link, page)
            text_content = parse_html_to_text(html_content)

            # Create a valid filename from the URL
            filename = link.replace(url, "").replace("/", "_").strip("_") + ".txt"
            file_path = output_path / filename

            # Save the content to a file
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_content)
        
        print(f"Documentation successfully saved to {output_folder}")
        await browser.close()

if __name__ == "__main__":
    url = input("Enter the URL of the documentation site: ")
    output_folder = input("Enter the output folder path: ")
    asyncio.run(save_documentation(url, output_folder))
