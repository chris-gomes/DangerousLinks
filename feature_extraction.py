# import libraries
import pandas as pd
import numpy as np
import url_and_whois_utils as utils
import json

from sklearn.preprocessing import LabelBinarizer

def url_feature_extraction(dataframe):
    """ Extracts the desired features from the URLs that are provided.
    Params:
        dataframe (Pandas Data Frame): list of URLs to extract from
    Returns:
        (Pandas Data Frame) the original data with features as new columns
    """
    df = dataframe.copy()

    # check if URL is less than 54 (0), between 54 and 75 (0.5), or greater than 75 (1)
    df["URL Length"] = np.where(df["URL"].str.len() < 54, 0, 0.5)
    df["URL Length"] = np.where(df["URL"].str.len() > 75, 1, df["URL Length"])
    print("Extracted: URL Length")

    # get length of Domain
    df["Domain Length"] = utils.min_max_norm(df["Domain"].str.len())
    print("Extracted: Domain Length")

    # check if bit.ly or tinyurl is used (1 means yes)
    df["Shortened"] = np.where(df["URL"].str.contains("bit.ly", case=False), 1, 0)
    df["Shortened"] = np.where(df["URL"].str.contains("tinyurl.com", case=False), 1, df["Shortened"])
    print("Extracted: Shorten")

    # check if URL contains "@" (1 means yes)
    df["Contains @"] = np.where(df["URL"].str.contains("@"), 1, 0)
    print("Extracted: Contains @")

    # check if the URL contains "//" anywhere besides the beginning (1 means yes)
    df["Contains //"] = np.where(df["URL without Protocol"].str.contains("//"), 1, 0)
    print("Extracted: Contains //")

    # check if the Domain contains "-" (1 means yes)
    df["Domain Contains -"] = np.where(df["Domain"].str.contains("-"), 1, 0)
    print("Extracted: Contains -")

    # check if the Domain contains "_" (1 means yes)
    df["Domain Contains _"] = np.where(df["Domain"].str.contains("_"), 1, 0)
    print("Extracted: Contains _")

    # check if the Domain contains "=" (1 means yes)
    df["Domain Contains ="] = np.where(df["Domain"].str.contains("="), 1, 0)
    print("Extracted: Contains =")

    # check if the Domain contains "?" (1 means yes)
    df["Domain Contains ?"] = np.where(df["Domain"].str.contains(r"\?"), 1, 0)
    print("Extracted: Contains ?")

    # check number of periods in the Domain
    df["Number of Periods"] = utils.min_max_norm(df["Domain"].str.count("."))
    print("Extracted: Number of Periods")

    # check if HTTPS is used (1 means yes)
    df["Uses HTTPS"] = np.where(df["Protocol"].str.lower() == "https", 1, 0)
    print("Extracted: Uses HTTPS")

    # check if "https" is in the domain (1 means yes)
    df["Domain Contains HTTPS"] = np.where(df["Domain"].str.contains("https", case=False), 1, 0)
    print("Extracted: Domain Contains HTTPS")

    # check percent of URL that is symbols
    symbols = [r"//", "/", r"\?", "=", ",", ";", r"\(", r"\)", r"\]", r"\+"]
    df["Number of Symbols"] = df["URL without Protocol"].str.count(":")
    for s in symbols:
        df["Number of Symbols"] = df["Number of Symbols"].add(df["URL without Protocol"].str.count(s))
    df["Percent Symbols"] = df["Number of Symbols"].divide(df["URL"].str.len())
    print("Extracted: Percent Symbols")

    # calculate the entropty of the URL
    vect_str_entropy = np.vectorize(utils.str_entropy)
    df["URL Entropy"] = utils.min_max_norm(vect_str_entropy(df["URL"]))
    print("Extracted: Entropy of URL")

    return df

def whois_feature_extraction(dataframe):
    """ Extracts the desired features from the WHOIS information that is provided.
    Params:
        dataframe (Pandas Data Frame): list of WHOIS information to extract from
    Returns:
        (Pandas Data Frame) the original data with features as new columns
    """
    df = dataframe.copy()

    # vectorize function to check if values are in WHOIS info
    vect_in_whois = np.vectorize(utils.in_whois)

    # check if domain has whois information
    vect_has_whois = np.vectorize(utils.has_whois)
    df["In WHOIS"] = vect_has_whois(df["WHOIS"])
    print("Extracted: In WHOIS")

    # check if WHOIS has an update date
    df["Has Update Date"] = vect_in_whois(df["WHOIS"], "updated_date")
    print("Extracted: Has Update Date")

    # check if WHOIS has creation date
    df["Has Creation Date"] = vect_in_whois(df["WHOIS"], "creation_date")
    print("Extracted: Has Creation Date")

    # get percent of whois info that is filled
    vect_whois_precent_filled = np.vectorize(utils.whois_percent_filled)
    df["WHOIS Percent Filled"] = vect_whois_precent_filled(df["WHOIS"])
    print("Extracted: WHOIS Percent Filled")

    # get length of the registrar
    vect_registrar_length = np.vectorize(utils.registrar_length)
    df["Registrar Length"] = utils.min_max_norm(vect_registrar_length(df["WHOIS"]))
    print("Extracted: Registrar Length")

    # check if WHOIS has registered emails
    df["Has Email"] = vect_in_whois(df["WHOIS"], "emails")
    print("Extracted: Has Email")

    # check if WHOIS has registered org
    df["Has Org"] = vect_in_whois(df["WHOIS"], "org")
    print("Extracted: Has Org")

    # get country from WHOIS
    vect_get_country = np.vectorize(utils.get_country)
    df["Country"] = vect_get_country(df["WHOIS"])
    lb = LabelBinarizer()
    countries = lb.fit_transform(df["Country"])
    df = df.join(pd.DataFrame(data=countries, columns=lb.classes_))
    print("Extracted: Country")
    
    return df


def main():
    # read in aggregated raw data
    all_data = pd.read_csv("raw_data.csv")
    print("Raw Data Read In")

    # change negative label from -1 to 0
    all_data["Malicious"] = np.where(all_data["Malicious"] == -1, 0, all_data["Malicious"])

    # convert WHOIS data to JSON
    all_data["WHOIS"] = np.where(all_data["WHOIS"].isnull(), "{}", all_data["WHOIS"])
    vect_to_json = np.vectorize(json.loads)
    all_data["WHOIS"] = vect_to_json(all_data["WHOIS"])

    # extract URL features
    all_data = url_feature_extraction(all_data)
    print("URL Features Extracted")

    # extract WHOIS features
    all_data = whois_feature_extraction(all_data)
    print("WHOIS Features Extracted")

    # export resulting dataframe to a csv
    all_data.to_csv("final_data.csv", index=False)
    print("Data Exported to CSV")

main()