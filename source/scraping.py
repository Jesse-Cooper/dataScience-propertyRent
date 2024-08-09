"""
@context
    * Scrapes raw rental property data from Domain
    * Scraper works on Domain and complies with `robots.txt` as of `2024-08-09`
        * Find current `robots.txt` at `https://www.domain.com.au/robots.txt`
    * There is a minimum delay of `REQUEST_DELAY` seconds between each request
      to reduce the load on Domain's servers
"""


import os
import pandas   as pd
import requests
import time

from bs4         import BeautifulSoup
from collections import defaultdict
from datetime    import date
from zipfile     import ZipFile


DIR_DATASETS = "datasets"
DIR_SAVE_RENT = "datasets/raw/{date}_vic_scrape_rent.csv"

# * Search for rental properties by postcode
# * Domain has a search page limit of `50` which does not cover all the
#   properties it offers
#     * Searching by postcode means more properties can be scraped
# * Although the program tries to grab only Victorian properties some locations
#   out of state share postcodes and may also be grabbed
URL = (
    "https://www.domain.com.au/rent/"
    "?postcode={postcode}"
    "&page={page}"
)

# * The majority of Victorian postcodes falls in/on this range
# * Some postcodes many not exist, but that does not cause any error
POSTCODE_MIN = 3000
POSTCODE_MAX = 3996

# * Domain has a max of `50` pages when searching
PAGES_TO_SCRAPE = 50

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(X11; Linux x86_64; rv:127.0) "
        "Gecko/20100101 "
        "Firefox/127.0"
    )
}

# * Elements of the parsed html to collect the features from
ELEMENT_PROPERTY_URL = "a"
ELEMENT_RENT         = "div"
ELEMENT_COORDINATES  = "a"

# * Unique identifying attributes of the elements to collect features from
ATTRS_PROPERTY_URL = {"class": r"address"}
ATTRS_RENT         = {"data-testid": "listing-details__summary-title"}
ATTRS_COORDINATES  = {"target": "_blank", "rel": "noopener noreferrer"}

# * Minimum delay between each request (in seconds)
REQUEST_DELAY = 1


def main():
    """
    @context
        * Scrapes and saves rental property data
        * Saves the data as a CSV
            * Finish date of the scrape is in the filename
        * Contains the features: `url`, `rent` and `coordinates`
        * Features are in raw text and need to be process
    """

    # * Unzip the dataset folder if it has not already been unzipped
    if not os.path.exists(DIR_DATASETS):
        print("Extracting zipped datasets")
        with ZipFile(f"{DIR_DATASETS}.zip", "r") as file:
            file.extractall(".")
        print("Extracted zipped datasets")

    urls = scrape_property_urls()
    data = scrape_property_data(urls)

    date_completed = "{:%Y_%m_%d}".format(date.today())

    # * Save scraped data as a CSV
    df = pd.DataFrame.from_dict(data)
    df.to_csv(DIR_SAVE_RENT.format(date=date_completed), index=False)

    print("Scraped data saved")


def scrape_property_urls():
    """
    @context
        * Scrapes unique rental property URLs from Domain by postcode

    @return : set[str]
        * Rental property URLs
    """

    print("Scraping rental property URLs")

    urls = set()

    for postcode in range(POSTCODE_MIN, POSTCODE_MAX + 1):
        for page in range(PAGES_TO_SCRAPE):

            # * Create a url for current `postcode` and `page` then request it
            # * Parse the raw html requested
            url = URL.format(postcode = postcode, page = page + 1)
            raw = requests.get(url, headers=HEADERS).text
            request_time = time.time()
            parsed = BeautifulSoup(raw, "html.parser")

            # * Page not loaded properly - skip it
            if not raw:
                continue

            # * Find all property urls on this page
            # * Move on to next the postcode if no property are found
            tags = parsed.find_all(ELEMENT_PROPERTY_URL, ATTRS_PROPERTY_URL)
            if not tags:
                break

            urls |= {tag["href"] for tag in tags}

            # * If there has been less than `REQUEST_DELAY` seconds from the
            #   last request, delay the next request by `REQUEST_DELAY` seconds
            if (time.time() - request_time) < REQUEST_DELAY:
                time.sleep(REQUEST_DELAY)

    print(f"Scraped {len(urls)} rental property URLs")

    return urls


def scrape_property_data(urls):
    """
    @context
        * Scrapes selected features from each rental property URL in `urls`
        * This function may take a very long time

    @parameters
        * urls : set[str]
            * Rental property URLs to scrape data from

    @return : dict[str, list[str]]
        * Dictionary organised into named lists like a table
        * Contains the features: `url`, `rent` and `coordinates`
        * Values at the same index for each list belong to the same property
    """

    print("Scraping data from rental property URLs")

    data = defaultdict(list)

    for url in urls:

        # * Request raw html text of `url` and parse it
        raw = requests.get(url, headers=HEADERS).text
        request_time = time.time()
        parsed = BeautifulSoup(raw, "html.parser")

        # * Page not loaded properly - skip it
        if not raw:
            continue

        # * Get tags of elements in `parsed` containing the features to scrape
        tag_rent = parsed.find(ELEMENT_RENT, ATTRS_RENT)
        tag_coordinates = parsed.find(ELEMENT_COORDINATES, ATTRS_COORDINATES)

        # * Some features for this property may not exist
        # * If any feature remains `None` this property is not saved
        rent = coordinates = None

        if tag_rent:
            rent = tag_rent.getText()

        if tag_coordinates:
            coordinates = tag_coordinates.attrs["href"]

        # * Do not add property if any features are missing
        if None in [rent, coordinates]:
            continue

        data["url"].append(url)
        data["rent"].append(rent)
        data["coordinates"].append(coordinates)

        # * If there has been less than `REQUEST_DELAY` seconds from the
        #   last request, delay the next request by `REQUEST_DELAY` seconds
        if (time.time() - request_time) < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY)

    print(f"Scraped data of {len(data["url"])} rental property URLs")

    return data


if __name__ == "__main__":
    main()
