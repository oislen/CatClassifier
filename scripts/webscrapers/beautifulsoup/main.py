import os
import time
import requests
import numpy as np
from urllib.parse import urljoin
from bs4 import BeautifulSoup

def scrape_urls(n_images, home_url, search):
    """"""
    # create list of urls for scraping
    urls = []
    skip = 100
    npages = int(np.ceil(n_images / skip))
    for i in range(npages):
        skip_param = str(i * skip)
        html_query = f'search/?q={search}&skip={skip_param}'
        url = urljoin(home_url, html_query)
        urls.append(url)
    return urls

def scrape_srcs(n_images, home_url, urls):
    """"""
    # parse image sources from urls
    image_cnt = 0
    for url in urls:
        srcs= []
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        root_table_level = soup.find("div", {"id":"spiccont"})
        sub_table_level = root_table_level.find_all("div", {"class":"imdiv"})
        for row in sub_table_level:
            src = home_url + row.find('img')['src']
            srcs.append(src)
            image_cnt = image_cnt + 1
            if image_cnt == n_images:
                return srcs

def download_srcs(srcs, delay = 3):
    """"""
    # download images
    for src in srcs:
        img_data = requests.get(src).content
        image_fname = src.split('/')[-1]
        image_fpath = os.path.join(output_dir, image_fname)
        with open(image_fpath, 'wb') as handler:
            handler.write(img_data)
        time.sleep(delay)
    return 0

# if running script as main programme
if __name__ == '__main__':
    
    # set script constants
    search = 'dogs'
    n_images = 10000
    output_dir = f'C:\\Users\\oisin\\Downloads\\{search}'
    home_url = 'https://free-images.com'

    # if output directory does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)

    # run function and scrape urls
    urls = scrape_urls(n_images, home_url, search)
    # run function and scrape srcs
    srcs = scrape_srcs(n_images, home_url, urls)
    # run function to download srcs
    download_srcs(srcs, delay = 3)
