# pip install requests
# pip install beautifulsoup4
# pip install playwright
# playwright install
# https://playwright.dev/python/docs/intro

import os
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import cons
  
# define main webscraping programme
def webscraper(search, n_images, image_size, output_dir, headless = True):

    # run playwright in a syncronised fashion
    with sync_playwright() as p:

        print('Checking output directory ...')
        # if output directory does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok = True)

        print('Loading programme constants ...')
        # create a counter for the number of pages pulled
        page_idx = cons.page_idx
        image_idx = cons.image_idx
        scrap_images = cons.scrap_images

        print('Launching Chrome ...')
        browser = p.chromium.launch(headless = headless)

        print('Opening new browser page ...')
        page = browser.new_page()

        print('Going to free-images ...')
        page.goto(cons.free_images_url)
        page.wait_for_load_state()

        print('Searching website ...')
        page.locator(cons.search_bar_selector).type(search, delay = 0.1)
        page.locator(cons.search_button_selector).click()
        page.wait_for_load_state()

        # extract page url
        page_url = page.url

        # while still downloading images
        while scrap_images:
            
            print(f'Working on page {page_idx} ...')
            # update url with number of skipped images
            skip = page_idx * cons.images_per_page
            mod_page_url = f'{page_url}&skip={skip}'
            # go to updated web page
            page.goto(mod_page_url)
            page.wait_for_load_state()

            print('Scraping hrefs for images ...')
            hrefs= []
            response = requests.get(mod_page_url)
            soup = BeautifulSoup(response.text, "html.parser")
            root_table_level = soup.find("div", {"id":"spiccont"})
            sub_table_level = root_table_level.find_all("div", {"class":"imdiv"})
            for row in sub_table_level:
                href = cons.free_images_url + row.find('a', href=True)['href']
                hrefs.append(href)
            
            # loop over urls and open in new pages
            for url in hrefs:
                print(f'Pulling image: {image_idx} ...')
                # open image in new page
                new_page = browser.new_page()
                new_page.goto(url)
                new_page.wait_for_load_state()
                # open download options
                new_page.locator(cons.download_selector).click()
                new_page.wait_for_load_state()

                # extract out the image filename
                a_class_html = new_page.wait_for_selector(cons.image_filename_selector).inner_html()
                img_fname = BeautifulSoup(a_class_html, "html.parser").find('a', href = True)['href'].split('/')[-1]

                # download image
                with new_page.expect_download() as download_info:
                    new_page.locator(cons.image_size_selector_dict[image_size]).click()
                    new_page.wait_for_load_state()
                download = download_info.value
                download.save_as(os.path.join(output_dir, img_fname))

                # if number of desired images has been collected
                if image_idx >= n_images:
                    print('Scraped all images ...')
                    scrap_images = False
                    new_page.close()
                    break

                # update image index
                image_idx = image_idx + 1
                new_page.close()
    
            # update image index
            page_idx = page_idx + 1
        
        print('Closing browser ...')
        browser.close()

# if running as main programme
if __name__ == '__main__':

    print('Starting programme ...')
    # execute main webscrapping programme
    search = 'cats'
    webscraper(
        search = search,
        n_images = 1000, 
        image_size = 'small', 
        output_dir = f'E:\\GitHub\\cat_classifier\\data\\{search}'
        )