# TODO: apply multiprocessing to for loops
import os
import requests
import logging
import numpy as np
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from multiprocessing import Pool
from beartype import beartype
import cons

@beartype
def gen_urls(search:str, n_images:int=cons.n_images, home_url:str=cons.home_url) -> list:
    """This function generates all relevant page urls for scraping
    
    Parameters
    ----------
    search : str
        The text to search and scrape for
    n_images : int
        The number of images to scrape, default is cons.n_images
    home_url : str
        The url of the home page to web scrape from, default is cons.home_url
    
    Returns
    -------
    list
        The urls to web scrape
    """
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

@beartype
def scrape_srcs(urls:list, n_images:int=cons.n_images, home_url:str=cons.home_url):
    """This function scrapes image scrs from the given urls
    
    Parameters
    ----------
    urls : list
        The image urls to scrape srcs for
    n_images : int
        The number of images to scrape, default is cons.n_images
    home_url : str
        The url of the home page to web scrape from, default is cons.home_url
    
    Returns
    -------
    list
        The scrape image srcs
    """
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

@beartype
def download_src(src:str, output_dir:str, search:str):
    """This function downloads a scraped image sources
    
    Parameters
    ----------
    src : str
        The image src to download to disk
    output_dir : str
        The output file directory to download the image srcs to
    search : str
        The text searched and scraped for
    
    Returns
    -------
    """
    img_data = requests.get(src).content
    image_fstem = src.split('/')[-1]
    image_fname = f'{search}.{image_fstem}'
    image_fpath = os.path.join(output_dir, image_fname)
    logging.info(image_fpath)
    with open(image_fpath, 'wb') as handler:
        handler.write(img_data)

@beartype
def multiprocess(func, args, ncpu:int=os.cpu_count()) -> list:
    """This utility function applyies another function in parallel given a specified number of cpus
    
    Parameters
    ----------
    func : function
        The function to run in parallel
    args : dict
        The arguments to pass to the function
    ncpu : int
        The number of cpus to use for parallel processing, default is os.cpu_count()
    
    Returns
    -------
    list
        The multiprocessing results
    """
    pool = Pool(ncpu)
    results = pool.starmap(func, args)
    pool.close()
    return results

@beartype
def webscraper(search:str, n_images:int=cons.n_images, home_url:str=cons.home_url, output_dir:str=cons.train_fdir):
    """The main beautiful soup webscrapping programme

    Parameters
    ----------
    search : str
        The text to search for
    n_images : int
        The number of images to web scrape, default is cons.n_images
    home_url : str
        The url for the home page to web scrape from, default is cons.home_url
    output_dir : str
        The output file directory to download the scraped images to, default is cons.train_fdir

    Returns
    -------
    """
    # if output directory does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)
    logging.info('Scraping urls ...')
    # run function and scrape urls
    urls = gen_urls(search=search, n_images=n_images, home_url=home_url)
    logging.info('Scraping srcs ...')
    # run function and scrape srcs
    srcs = scrape_srcs(urls=urls, n_images=n_images, home_url=home_url)
    # run function to download src
    multiprocess(download_src, [(src, output_dir, search) for src in srcs])