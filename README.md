# DangerousLinks

The goal of this project is to develop and test machine learning models that could be used to determine if a site is malicious based off its URL and WHOIS information.

Datasets Used: https://www.unb.ca/cic/datasets/url-2016.html

Python Library for WHOIS: https://pypi.org/project/whois/#description

Publications Referenced for Feature Extraction:
  1. https://cseweb.ucsd.edu/~voelker/pubs/mal-url-kdd09.pdf
  2. https://ieeexplore.ieee.org/document/6691627
  3. https://archive.ics.uci.edu/ml/datasets/Phishing+Websites#
  4. https://link.springer.com/chapter/10.1007/978-3-319-46298-1_30#Sec6

Structure of Directory:
* feature_extraction.py - the script that was run to create the engineered features that were used in the models
* models.py - code for training and testing the models for this project
* raw_data_agg.py - this is the Python script that was run to aggregate together all of the original URL and WHOIS info (takes a long time to run since it has to search WHOIS for every URL)
* url_and_whois_utils.py - this is a collection of functions to make collecting the data and creating the features easier and cleaner
