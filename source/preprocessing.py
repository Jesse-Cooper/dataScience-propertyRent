"""
@context
    * Preprocesses cleaned rental property data and other support datasets
    * Rental property data must be cleaned before preprocessing
    * Preprocessed data is complete with no missing values
"""


import geopandas as gpd
import os
import pandas    as pd

from zipfile import ZipFile


DIR_DATASETS = "datasets"

# * Support datasets
DIR_WORK_HOURS = "datasets/raw/2021_vic_sa2_abs_census_work_hours.csv"
DIR_INCOME     = "datasets/raw/2021_vic_sa2_abs_census_income.csv"
DIR_TRANSPORT  = "datasets/raw/2021_vic_sa2_abs_census_transport.csv"
DIR_SA2        = "datasets/raw/2021_vic_sa2_geometry"

DIR_RENT = "datasets/curated/vic_latLon_scrape_rent"

DIR_SAVE_PREPROCESSED = "datasets/curated/preprocessed.csv"

# * The order and what to rename features of each dataset
# * All other features in the datasets are dropped
COLS_KEEP_WORK_HOURS = {
    "sa2_code_2021": "SA2",
    "p_tot_tot":     "population",
    "p_tot_1_19":    "work_hours_1_to_19",
    "p_tot_20_29":   "work_hours_20_to_29",
    "p_tot_30_34":   "work_hours_30_to_34",
    "p_tot_35_39":   "work_hours_35_to_39",
    "p_tot_40_44":   "work_hours_40_to_44",
    "p_tot_45_49":   "work_hours_45_to_49",
    "p_tot_50_over": "work_hours_over_49"
}
COLS_KEEP_INCOME = {
    "sa2_code_2021":             "SA2",
    "median_tot_hhd_inc_weekly": "median_household_income"
}
COLS_KEEP_TRANSPORT = {
    "sa2_code_2021":                 "SA2",
    "tot_p":                         "population",
    "one_method_car_as_passenger_p": "transport_only_car_passenger",
    "one_method_car_as_driver_p":    "transport_only_car_driver",
    "worked_home_p":                 "worked_home",
    "one_method_tot_one_method_p":   "transport_methods_1"
}
COL_KEEP_RENAME_RENT = {
    "sa2_code": "SA2",
    "rent":     "rent"
}

# * Data wrangles of each dataset
# * All features in each list is summed to create the new features
WRANGLES_WORK_HOURS = {
    "employed": [
        "work_hours_1_to_19",
        "work_hours_20_to_29",
        "work_hours_30_to_34",
        "work_hours_35_to_39",
        "work_hours_40_to_44",
        "work_hours_45_to_49",
        "work_hours_over_49"
    ]
}
WRANGLES_TRANSPORT = {
    "transport_only_car": [
        "transport_only_car_passenger",
        "transport_only_car_driver"
    ]
}

# * Columns to element-wise divide by their corresponding SA2 zone's population
POPULATION_WORK_HOURS = [
    "employed"
]
POPULATIONS_TRANSPORT = [
    "transport_methods_1",
    "transport_only_car",
    "worked_home"
]

FINAL_VARIABLES = [
    "mean_rent",
    "employed",
    "median_household_income",
    "transport_methods_1",
    "transport_only_car",
    "worked_home"
]

# * Outliers are outside of the whiskers are dropped
BOXPLOT_WHISKER_LENGTH = 1.5


def main():
    """
    @context
        * Preprocesses the cleaned rental property data and other support
          datasets
        * Saves the data as a CSV file
    """

    # * Unzip the dataset folder if it has not been already
    if not os.path.exists(DIR_DATASETS):
        print("Extracting zipped datasets")
        with ZipFile(f"{DIR_DATASETS}.zip", "r") as file:
            file.extractall(".")
        print("Extracted zipped datasets")

    df_rent = load_property_data()

    if type(df_rent) is not pd.DataFrame and df_rent == None:
        print("Need to clean rental property data before preprocessing")
        return

    # * Collate all support datasets
    df_work_hours = load_support_data(DIR_WORK_HOURS, COLS_KEEP_WORK_HOURS)
    df_income     = load_support_data(DIR_INCOME, COLS_KEEP_INCOME)
    df_transport  = load_support_data(DIR_TRANSPORT, COLS_KEEP_TRANSPORT)

    # * Do outlier detection on individual properties before aggregating them
    #   into their SA2 zones
    n_before = df_rent.shape[0]
    df_rent = outliers_boxplot(df_rent, "rent")
    df_rent = df_rent.dropna(how="any")
    n_after = df_rent.shape[0]

    print(
        f"{n_before - n_after} of {n_before} properties dropped for being "
        "outliers\n"
    )

    # * Wrangle new features
    df_work_hours = wrangle_support(df_work_hours, WRANGLES_WORK_HOURS)
    df_transport  = wrangle_support(df_transport, WRANGLES_TRANSPORT)
    df_rent = wrangle_property(df_rent)

    # * Standardise some feature
    df_work_hours = standardise_population(df_work_hours, POPULATION_WORK_HOURS)
    df_transport = standardise_population(df_transport, POPULATIONS_TRANSPORT)

    # * Join all dataset on SA2 zones
    df = join_dfs([df_work_hours, df_income, df_transport, df_rent])
    df = df[FINAL_VARIABLES]

    # * Mark and remove outliers column-wise
    # * Missing values do not count as outliers but are also removed here
    n_before = df.shape[0]
    n_nan_before = (df.isna().sum(axis=1) > 0).sum()
    for col in FINAL_VARIABLES:
        df = outliers_boxplot(df, col)
    df = df.dropna(how="any")
    n_after = df.shape[0]

    print(
        f"{n_nan_before} of {n_before} SA2 zones dropped for having missing "
        "values"
    )

    print(
        f"{n_before - n_nan_before - n_after} of {n_before - n_nan_before} SA2 "
        "zones dropped for being outliers\n"
    )

    # * Save preprocessed data as a CSV file
    df.to_csv(DIR_SAVE_PREPROCESSED, index=False)
    print(f"Preprocessed data saved with {n_after} SA2 zones")


def load_property_data():
    """
    @context
        * Loads cleaned rental property dataset
        * Each property is assigned an SA2 zone its coordinates fall in
        * Only the features wanted are kept being renamed and ordered

    @returns : Dataframe | None
        * Property dataset
        * `None` if cleaned dataset does not exist
    """

    if not os.path.exists(DIR_RENT):
        return None

    sdf_sa2 = gpd.read_file(DIR_SA2)
    sdf_sa2 = sdf_sa2.set_crs(epsg=4326)
    sdf_rent = gpd.read_file(DIR_RENT)

    # * Place each property in their appropriate SA2 zone
    # * If a property does not land in an SA2 zone they are removed
    n_before = sdf_rent.shape[0]
    df = gpd.sjoin(sdf_rent, sdf_sa2)
    n_after = df.shape[0]

    print(
        f"{n_before - n_after} of {n_before} properties dropped for being "
        f"outside all Victorian SA2 zones\n"
    )

    # * Remove all but the columns listed and rename/order them
    df = df[COL_KEEP_RENAME_RENT.keys()]
    return df.rename(columns=COL_KEEP_RENAME_RENT)


def load_support_data(dir, cols_keep):
    """
    @context
        * Loads a support dataset at the directory `dir`

    @parameters
        * dir : str
            * Directory of dataset to load
        * cols_keep : dict[str, str]
            * The order and what to rename features
            * All other features in the dataset are dropped

    @return : Dataframe
        * Dataset at the directory `dir` with the required feature set
    """

    df = pd.read_csv(dir)

    # * Remove all but the columns listed and rename/order them
    df = df[cols_keep.keys()]
    df = df.rename(columns=cols_keep)

    return df


def wrangle_support(df, wrangles):
    """
    @context
        * Creates new features in `df` by summing the existing features in
          `wrangles`

    @parameters
        * df : Dataframe
            * Dataset to create new feature in from existing features
        * wangles : dict[str, list[str]]
            * New features to create by summing existing features
            * Key - what to name the new feature
            * Value - list of the existing feature names to sum

    @return : Dataframe
        * Dataset with the wrangled features
    """
    for name, features in wrangles.items():
        df[name] = df[features].sum(axis=1, skipna=True)
    return df


def wrangle_property(df):
    """
    @context
        * Creates the new feature `mean_rent` by aggregating the rent of
          properties by their SA2 zone in `df`

    @parameters
        * df : Dataframe
            * Dataset of rental properties

    @return : Dataframe
        * Dataset of properties grouped into their SA2 zones to get the mean of
          the rent
    """
    df = df.groupby("SA2", as_index=False)["rent"].mean()
    return df.rename(columns={"rent": "mean_rent"})


def standardise_population(df, cols):
    """
    @context
        * For each feature in `cols` divide each value by their corresponding
          SA2 zone's population
        * Standardised to `[0, 1]`
        * Cells with `None` stay as `None`

    @parameters
        * df : Dataframe
            * Dataset to standardise features of
        * cols : list[str]
            * Names of columns to standardise

    @return : Dataframe
        * Dataset of features standardised
    """
    df[cols] = df[cols].div(df["population"].values, axis=0)
    return df


def join_dfs(dfs):
    """
    @context
        * Outer joins all datasets in `dfs` on their SA2 zones
        * Outer join as we want to keep as much data for outlier detection as
          possible - missing data later dropped

    @parameters
        * dfs : list[Dataframe]
            * Datasets to join together
            * Must all have the feature: `SA2`

    @return : Dataframe
        * Dataset of all datasets joined together
    """

    # * Must have at least 2 datasets in `dfs`
    assert(len(dfs) >= 2)

    # * All datasets are outer joined on their SA2 zones
    df_merge = dfs[0]
    for df in dfs[1:]:
        df_merge = df_merge.merge(df, on="SA2", how="outer")

    return df_merge


def outliers_boxplot(df, on):
    """
    @context
        * Marks outlier of feature `on` in `df` defined by a boxplot
        * Cells that are outliers for feature `on` are replaced with `None`

    @parameters
        * df : Dataframe
            * Dataset containing `on` to find outliers in
        * on : str
            * Name of the feature to find outliers in

    @return : Dataframe
        * Dataset with outliers of feature `on` in `df` marked
    """

    # * Values that define the bounds of outliers in a boxplot
    # * Does not count any missing values
    q1 = df[on].quantile(0.25)
    q3 = df[on].quantile(0.75)
    iqr = q3 - q1

    n_before = df[on].shape[0]
    n_nan_before = df[on].isna().sum()

    # * Mark outlier cells by replacing them with `None`
    lower_bounds = q1 - (BOXPLOT_WHISKER_LENGTH * iqr)
    upper_bounds = q3 + (BOXPLOT_WHISKER_LENGTH * iqr)
    df.loc[~df[on].between(lower_bounds, upper_bounds), on] = None

    n_nan_after = df[on].isna().sum()

    print(
        f"{n_nan_after - n_nan_before:>3} of {n_before - n_nan_before:>3} "
        f"outliers marked in {on}"
    )

    return df


if __name__ == "__main__":
    main()
