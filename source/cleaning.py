"""
@context
    * Cleans the most recently scraped rental property data
    * Data must be scraped before cleaning
"""


import geopandas as gpd
import os
import pandas    as pd
import re

from shapely.geometry import Point
from zipfile          import ZipFile


DIR_DATASETS = "datasets"
DIR_RAW_RENT = "datasets/raw"
FILENAME_RAW_RENT = "vic_scrape_rent.csv"

DIR_SAVE_CURATED_RENT = (
    "datasets/curated/vic_latLon_scrape_rent/vic_latLon_scrape_rent.shp"
)

# * Numbers in rent text with these after them are excluded
# * Only want rent per week
# * Some are likely to slip through, but these exclusions should catch most
EXCLUSIONS = [
    "calendar month",
    "cm",
    "m2",
    "month",
    "p/m",
    "p.a",
    "p.c.m",
    "pcm",
    "per calendar month",
    "per fortnight",
    "per month",
    "per night",
    "per year",
    "pm"
]

# * Regex pattens to extract features from the raw text
RE_COORDINATES = r"destination=([\d\.\-]+),([\d\.\-]+)"
RE_RENT = r"(\d[\d\,\.]+)(?![\s\,\.\/]*(" + "|".join(EXCLUSIONS) + r"|\d))"


def main():
    """
    @context
        * Cleans and save the most recently scraped rental property data
        * Saves the data as a shapefile
        * Contains the features: `rent` and `geometry`
    """

    # * Unzip the dataset folder if it has not already been unzipped
    if not os.path.exists(DIR_DATASETS):
        print("Extracting zipped datasets")
        with ZipFile(f"{DIR_DATASETS}.zip", "r") as file:
            file.extractall(".")
        print("Extracted zipped datasets")

    # * Get the filename of most recently data scraped
    filename = find_raw_rent_filename()
    if filename == None:
        print("Need to scrape rental property data before cleaning")
        return

    df_raw = pd.read_csv(f"{DIR_RAW_RENT}/{filename}")

    # * Clean the raw features of `df_raw`
    df_curated = pd.DataFrame()
    df_curated["rent"] = df_raw["rent"].apply(extract_rent)
    df_curated["geometry"] = df_raw["coordinates"].apply(extract_geometry)

    # * Properties with no geometry or rent have no use in this project
    n_before = df_curated.shape[0]
    df_curated = df_curated.dropna(how="any")
    n_after = df_curated.shape[0]

    print(f"Lost {n_before - n_after} / {n_before} properties from cleaning")

    # * Save cleaned data as a shapefile
    sdf_curated = gpd.GeoDataFrame(df_curated, geometry=df_curated["geometry"])
    sdf_curated = sdf_curated.set_crs(epsg=4326)
    sdf_curated.to_file(
        DIR_SAVE_CURATED_RENT,
        driver="ESRI Shapefile",
        index=False
    )

    print(f"Saved {n_after} cleaned properties")


def find_raw_rent_filename():
    """
    @context
        * Finds the filename of the most recent scraped data
        * Files should have the scraping date at the start of their filename

    @return : str | None
        * Filename of the most recent scraped data
        * `None` if the file does not exist
    """

    # * Find files in `DIR_RAW_RENT` with `FILENAME_RAW_RENT` in its filename
    name_files = [
        file
        for file in os.listdir(DIR_RAW_RENT)
        if FILENAME_RAW_RENT in file
    ]

    if len(name_files) == 0:
        return None

    return sorted(name_files, reverse=True)[0]


def extract_rent(raw_rent_text):
    """
    @context
        * Extracts the weekly rent from `raw_rent_text`

    @parameters
        * raw_rent_text : str
            * Raw text to extract the weekly rent from

    @return : int | None
        * Weekly rent extracted from `raw_rent_text`
        * Rents are only integers, any decimal part is dropped
        * `None` if the rent cannot be extracted
    """
    match = re.search(RE_RENT, raw_rent_text, re.IGNORECASE)
    if match:
        return int(float(match.group(1).replace(",", "")))
    return None


def extract_geometry(raw_coordinate_text):
    """
    @context
        * Extracts the geometry from `raw_coordinate_text`

    @parameters
        * raw_coordinate_text : str
            * Raw text to extract the latitude and longitude from

    @return : Point | None
        * Geometry extracted from `raw_coordinate_text`
        * Point ordered longitude then latitude as that is how geopandas orders
          its geometry
        * `None` if neither latitude or longitude can be extracted
    """

    match = re.search(RE_COORDINATES, raw_coordinate_text)

    if match and len(match.groups()) == 2:
        latitude = float(match.group(1))
        longitude = float(match.group(2))

        return Point(longitude, latitude)

    return None


if __name__ == "__main__":
    main()
