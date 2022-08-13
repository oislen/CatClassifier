import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# set script parameters
search = 'dogs'
n_images = 5
output_dir = f'C:\\Users\\oisin\\Downloads\\{search}'

# if output directory does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok = True)

# create list of urls for scraping
urls = []
skip = 100
home_url = 'https://free-images.com'
for i in range(1):
    skip_param = str(i * skip)
    html_query = f'search/?q={search}&skip={skip_param}'
    url = urljoin(home_url, html_query)
    urls.append(url)

# parse image sources from urls
for url in urls:
    srcs= []
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    root_table_level = soup.find("div", {"id":"spiccont"})
    sub_table_level = root_table_level.find_all("div", {"class":"imdiv"})
    for row in sub_table_level:
        src = home_url + row.find('img')['src']
        srcs.append(src)

# download images
for src in srcs[:n_images]:
    img_data = requests.get(src).content
    image_fname = src.split('/')[-1]
    image_fpath = os.path.join(output_dir, image_fname)
    with open(image_fpath, 'wb') as handler:
        handler.write(img_data)
