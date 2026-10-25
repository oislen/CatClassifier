free_images_url = "https://free-images.com"

images_per_page = 100
page_idx = 0
image_idx = 1
scrap_images = True

image_selector_default = "#ol > div > table > tbody > tr:nth-child(2) > td:nth-child(4)"

image_size_selector_dict = {
    "tiny": "#ol > div > table > tbody > tr:nth-child(1) > td:nth-child(4)",
    "small": image_selector_default,
    "medium": image_selector_default,
}

search_bar_selector = (
    "#sbar > form > table > tbody > tr > td:nth-child(1) > input[type=text]"
)
search_button_selector = (
    "#sbar > form > table > tbody > tr > td:nth-child(3) > button > i"
)
download_selector = "#imwrp #timg"
image_filename_selector = (
    "#ol > div > table > tbody > tr:nth-child(2) > td:nth-child(4)"
)
