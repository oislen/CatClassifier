import unittest
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "webscrapers"))

import cons
from utilities.webscraper import scrape_srcs

exp_srcs = ['https://free-images.com/sm/d790/cat_home_cat_looking.jpg', 'https://free-images.com/sm/e396/cat_hangover_red_cute_51.jpg', 'https://free-images.com/sm/bc44/cat_hangover_siamese_cat_2.jpg']
urls = ['https://free-images.com/search/?q=cat&skip=0']
obs_srcs = scrape_srcs(urls=urls, n_images=3, home_url=cons.home_url)

class Test_scrape_srcs(unittest.TestCase):
    """"""

    def setUp(self):
        self.obs_srcs = obs_srcs
        self.exp_srcs = exp_srcs

    def test_type(self):
        self.assertEqual(type(self.obs_srcs), type(self.exp_srcs))

    def test_len(self):
        self.assertEqual(len(self.obs_srcs), len(self.exp_srcs))

    def test_values(self):
        self.assertEqual(self.obs_srcs, self.exp_srcs)

if __name__ == "__main__":
    unittest.main()
