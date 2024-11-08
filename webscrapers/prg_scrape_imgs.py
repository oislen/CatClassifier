import logging
import cons
from commandline_interface import commandline_interface
from webscrapers.download_comp_data import download_comp_data
from webscrapers.beautifulsoup.webscraper import webscraper
from beartype import beartype

@beartype
def scrape_imags(
    run_download_comp_data:bool=False, 
    run_webscraper:bool=False
    ):
    """Programme for running Kaggle comp data download and image web scrapers

    Parameters
    ----------
    run_download_comp_data : bool
        Whether to run the download Kaggle competition data, default is False
    run_webscraper : bool
        Whether to run the image webscraper, default is False

    Returns
    -------
    """
    if run_download_comp_data:
        logging.info('Downloading kaggle data ...')
        # download competition data
        download_comp_data(
            comp_name=cons.comp_name,
            data_dir=cons.data_fdir,
            download_data=cons.download_data, 
            unzip_data=cons.unzip_data, 
            del_zip=cons.del_zip
            )
    if run_webscraper:
        logging.info('Running cat image webscraper ...')
        # run cat webscraper
        webscraper(
            search='cat', 
            n_images=cons.n_images, 
            home_url=cons.home_url, 
            output_dir=cons.train_fdir
            )
        logging.info('Running dog image webscraper ...')
        # run dog webscraper
        webscraper(
            search='dog', 
            n_images=cons.n_images, 
            home_url=cons.home_url, 
            output_dir=cons.train_fdir
            )

# if running as main programme
if __name__ == '__main__':
    
    # set up logging
    lgr = logging.getLogger()
    lgr.setLevel(logging.INFO)

    # handle input parameters
    input_params_dict = commandline_interface()

    # run the scrape images programme
    scrape_imags(
        run_download_comp_data=input_params_dict['run_download_comp_data'], 
        run_webscraper=input_params_dict['run_webscraper']
        )