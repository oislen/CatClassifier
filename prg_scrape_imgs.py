import cons
from webscrapers.beautifulsoup.main import main

# if running as main programme
if __name__ == '__main__':
    
    # set number of images constants
    n_images = 100
    # set search constants
    cats_search = 'cats'
    dogs_search = 'dogs'
    # set output directory constants
    cat_output_dir = cons.output_dir.format(search = cats_search)
    dog_output_dir = cons.output_dir.format(search = dogs_search)

    # run main function
    main(search = cats_search, n_images = n_images, home_url = cons.home_url, output_dir = cat_output_dir)
    main(search = dogs_search, n_images = n_images, home_url = cons.home_url, output_dir = dog_output_dir)