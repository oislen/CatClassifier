# TODO: apply multiprocessing to for loops
import os
import requests
import logging
import numpy as np
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from multiprocessing import Pool

def gen_urls(n_images, home_url, search):
    """
    This function generates all relevant page urls for scraping
    """
    logging.info('Scraping urls ...')
    # create list of urls for scraping
    urls = []
    skip = 100
    npages = int(np.ceil(n_images / skip))
    for i in range(npages):
        logging.info(i)
        skip_param = str(i * skip)
        html_query = f'search/?q={search}&skip={skip_param}'
        url = urljoin(home_url, html_query)
        urls.append(url)
    return urls

def scrape_srcs(n_images, home_url, urls):
    """
    This function scrapes image scrs from the given urls
    """
    logging.info('Scraping srcs ...')
    # parse image sources from urls
    image_cnt = 0
    srcs= []
    for url in urls:
        logging.info(url)
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

def download_src(src, output_dir, search):
    """
    This function downloads a scraped image sources
    """
    img_data = requests.get(src).content
    image_fstem = src.split('/')[-1]
    image_fname = f'{search}.{image_fstem}'
    image_fpath = os.path.join(output_dir, image_fname)
    logging.info(image_fpath)
    with open(image_fpath, 'wb') as handler:
        handler.write(img_data)
    return 0

def multiprocess(func, args, ncpu = os.cpu_count()):
    """
    This utility function applyies another function in parallel given a specified number of cpus
    """
    pool = Pool(ncpu)
    results = pool.starmap(func, args)
    pool.close()
    return results

def main(search, n_images, home_url, output_dir):
    """
    """
    # if output directory does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)
    # run function and scrape urls
    urls = gen_urls(n_images, home_url, search)
    # run function and scrape srcs
    srcs = scrape_srcs(n_images, home_url, urls)
    # run function to download src
    multiprocess(download_src, [(src, output_dir, search) for src in srcs])
    return 0