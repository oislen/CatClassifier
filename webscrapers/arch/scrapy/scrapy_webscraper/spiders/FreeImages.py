# scrapy crawl FreeImages

# import relevant libraries
from urllib.parse import urljoin
import scrapy
import sys
sys.path.append('E:\\GitHub\\\cat_classifier\\webscrapers\\scrapy_webscraper')
from scrapy_webscraper.items import ImageItem

class FreeImages(scrapy.Spider):
    """"""
    # set scraper name and allowed domains
    name = 'FreeImages'
    allowed_domains = ['free-images.com']
    # create start urls list for parsing
    start_urls = []
    q_param = 'dogs'
    skip = 100
    home_url = 'https://free-images.com'
    for i in range(1):
        skip_param = str(i * skip)
        html_query = f'search/?q={q_param}&skip={skip_param}'
        url = urljoin(home_url, html_query)
        start_urls.append(url)
    
    def parse(self, response):
        # loop over the imdiv classes divs
        for imdiv in response.xpath('//*[@id="spiccont"]/div[@class="imdiv"]'):
            # extract out image src
            img_url = imdiv.xpath(".//a/img/@src").extract_first()
            # create full url path to download image
            full_img_url = urljoin(self.home_url, img_url)
            yield ImageItem(image_urls=[full_img_url])
