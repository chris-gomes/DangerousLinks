import whois
import datetime
import math

def get_whois_info(x):
    """ Gets WHOIS information from WHOIS API for given URL domain
    Params:
        x (str): the URL
    Returns:
        (Dict) WHOIS information
    """
    try:
        whois_resp = whois.whois(x)
    except:
        return None
    print(x)
    return whois_resp

def str_entropy(x):
    """ Calculates the entropy of the given string.
    Params:
        x (str): string to calculate the entropy
    Returns:
        (number) the entropy of the string
    """
    unique_chars = dict()
    for char in x:
        if char in unique_chars:
            unique_chars[char] += 1
        else:
            unique_chars[char] = 1
    result = 0
    total_len = len(x)
    for char, count in unique_chars.items():
        prob = count / total_len
        result += (prob * math.log(prob, 2))
    return (-1 * result)

def min_max_norm(x):
    """ Min-Max normalizes the list of numbers.
    Params:
        x (Pandas Series): List of numbers. 
    Returns:
        (Pandas Series) List of numbers normalized between 0 and 1.
    """
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)

def has_whois(x):
    """ Determines if the given struct of WHOIS information is null or empty.
    Params:
        x (Dict): nested structure of WHOIS information
    Returns:
        (int) 0 for false and 1 for true 
    """
    all_none = True
    for key, value in x.items():
        if value != None:
            all_none = False
    if all_none:
        return 0
    else:
        return 1

def whois_percent_filled(x):
    """ Determines the precent of the struct of WHOIS information that is filled.
    Params:
        x (Dict): nested structure of WHOIS information
    Returns:
        (number) the percent of filled parts
    """
    not_null = 0
    for key, value in x.items():
        if value != None:
            not_null += 1
    if len(x) == 0:
        return 0
    else:
        return not_null / len(x)

def registrar_length(x):
    """ Finds the length of the registrar name in the WHOIS information struct.
    Params:
        x (Dict): nested structure of WHOIS information
    Returns:
        (int) the length of the registrar name
    """
    if x == None:
        return 0
    elif "registrar" in x and isinstance(x["registrar"], str):
        return len(x["registrar"])
    else:
        return 0

def in_whois(x, label):
    """ Determines if there is information at the given label of the WHOIS information struct.
    Params:
        x (Dict): nested structure of WHOIS information
        label (str): name of the nested information to check for
    Returns:
        (int) 0 for false and 1 for true
    """
    if label in x and x[label] != None:
        return 1
    else:
        return 0

def get_country(x):
    """ Gets the registered country in the WHOIS information struct.
    Params:
        x (Dict): nested structure of WHOIS information
    Returns:
        (str) the name of the country in lowercase
    """
    if x == None:
        return "no country"
    elif "country" in x and isinstance(x["country"], str):
        return x["country"].lower()
    else:
        return "no country"
