# pip install requests
# pip install beautifulsoup4
# pip install playwright
# playwright install
# https://playwright.dev/python/docs/intro

import requests
import time
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
  
def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # got to free-images website
        free_images_url = 'https://free-images.com'
        page.goto(free_images_url)
        page.wait_for_load_state()

        # search for cats
        page.locator('#sbar > form > table > tbody > tr > td:nth-child(1) > input[type=text]').type('cats', delay = 0.1)
        page.locator('#sbar > form > table > tbody > tr > td:nth-child(3) > button > i').click()
        page.wait_for_load_state()

        # parse urls to webpages with beautiful soup
        hrefs= []
        response = requests.get(page.url)
        soup = BeautifulSoup(response.text, "html.parser")
        root_table_level = soup.find("div", {"id":"spiccont"})
        sub_table_level = root_table_level.find_all("div", {"class":"imdiv"})
        for row in sub_table_level:
            href = free_images_url + row.find('a', href=True)['href']
            hrefs.append(href)
        
        # loop over urls and open in new pages
        for url in hrefs[0:3]:
            # open image in new page
            new_page = browser.new_page()
            new_page.goto(url)
            new_page.wait_for_load_state()
            # open download options
            new_page.locator('#imwrp #timg').click()
            new_page.wait_for_load_state()

            # extract out the image filename
            a_class_html = new_page.wait_for_selector('#ol > div > table > tbody > tr:nth-child(2) > td:nth-child(4)').inner_html()
            img_fname = BeautifulSoup(a_class_html, "html.parser").find('a', href = True)['href'].split('/')[-1]

            # download image
            with new_page.expect_download() as download_info:
                new_page.locator('#ol > div > table > tbody > tr:nth-child(2) > td:nth-child(4) > a').click()
                new_page.wait_for_load_state()
            download = download_info.value
            download.save_as(f"C:\\Users\\oisin\\Pictures\\{img_fname}")

            new_page.close()

        browser.close()
  
if __name__ == '__main__':
    main()