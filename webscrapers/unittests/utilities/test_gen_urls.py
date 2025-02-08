import unittest
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "webscrapers"))

import cons
from utilities.webscraper import gen_urls

exp_urls = ['https://free-images.com/search/?q=cat&skip=0', 'https://free-images.com/search/?q=cat&skip=100', 'https://free-images.com/search/?q=cat&skip=200']
obs_urls = gen_urls(search="cat", n_images=300, home_url=cons.home_url)

class Test_gen_urls(unittest.TestCase):
    """"""

    def setUp(self):
        self.obs_urls = obs_urls
        self.exp_urls = exp_urls

    def test_type(self):
        self.assertEqual(type(self.obs_urls), type(self.exp_urls))

    def test_len(self):
        self.assertEqual(len(self.obs_urls), len(self.exp_urls))

    def test_values(self):
        self.assertEqual(self.obs_urls, self.exp_urls)

if __name__ == "__main__":
    unittest.main()
