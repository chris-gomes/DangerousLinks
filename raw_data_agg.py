# import libraries
import pandas as pd
import numpy as np
import glob
import os
import url_and_whois_utils as utils

def aggregate_url_data():
    """ Reads in all of the CSVs that contain URLs and combines and labels the URLs.
    Params:
        None
    Returns:
        (Pandas Data Frame) combined URL information and labels
    """

    # read in benign dataset
    url_info = pd.read_csv("URL/Benign_list_big_final.csv", header=None, names=["URL"])
    url_info["Malicious"] = -1

    # read in the other datasets
    file_names = glob.glob(os.path.join("URL/", "*.csv"))
    for f in file_names:
        if "Benign" not in f:
            curr_data = pd.read_csv(f, header=None, names=["URL"])
            curr_data["Malicious"] = 1
            url_info = url_info.append(curr_data)

    # remove "www." form the URL
    url_info["URL"] = url_info["URL"].str.replace("www.", "")

    # get protocol (HTTP or HTTPS)
    url_info["Protocol"] = url_info["URL"].str.slice(stop=5)
    url_info["Protocol"] = url_info["Protocol"].str.rstrip(":")

    # get URL minus the protocol
    url_info["URL without Protocol"] = url_info["URL"].str.lstrip("http:// https://")

    # get domain
    url_split = url_info["URL without Protocol"].str.split("/", n=1, expand=True)
    url_info["Domain"] = url_split[0]

    print("URL Data Aggregated")

    return url_info

def whois_info(df):
    """ Uses the WHOIS API to get domain information for given list of URLs.
    Params:
        df (Pandas Data Frame): URLs and related information
    Returns:
        (Pandas Data Frame) original data with WHOIS information asa  new column
    """

    # get all of the whois data for the domain
    print("Starting WHOIS Data Collection...")
    df["WHOIS"] = df["URL"].apply(utils.get_whois_info)
    print("WHOIS Data Collected")

    return df

def main():
    # read in the CSVs of URL data
    all_data = aggregate_url_data()

    # get WHOIS data
    all_data = whois_info(all_data)

    # export resulting dataframe to a csv
    all_data.to_csv("raw_data.csv", index=False)

main()