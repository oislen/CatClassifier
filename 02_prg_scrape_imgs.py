import logging
import cons
from webscrapers.beautifulsoup.main import main

# if running as main programme
if __name__ == '__main__':
    
    # set up logging
    lgr = logging.getLogger()
    lgr.setLevel(logging.INFO)
    
    # run main function
    main(search='cat', n_images=cons.n_images, home_url=cons.home_url, output_dir=cons.train_fdir)
    main(search='dog', n_images=cons.n_images, home_url=cons.home_url, output_dir=cons.train_fdir)