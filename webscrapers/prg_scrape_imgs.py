import logging
import cons
from webscrapers.download_comp_data import download_comp_data
from webscrapers.beautifulsoup.webscraper import webscraper

# if running as main programme
if __name__ == '__main__':
    
    # set up logging
    lgr = logging.getLogger()
    lgr.setLevel(logging.INFO)

    # download competition data
    download_comp_data(
        comp_name=cons.comp_name,
        data_dir=cons.data_fdir,
        download_data=cons.download_data, 
        unzip_data=cons.unzip_data, 
        del_zip=cons.del_zip
        )

    # run main function
    webscraper(search='cat', n_images=cons.n_images, home_url=cons.home_url, output_dir=cons.train_fdir)
    webscraper(search='dog', n_images=cons.n_images, home_url=cons.home_url, output_dir=cons.train_fdir)