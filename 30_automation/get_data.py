# General data processing
import numpy as np
import pandas as pd

# Packages from scraping
from selenium import webdriver
from bs4 import BeautifulSoup

# For miscellaneous processes
import re       
import sys
import joblib
import ast

def get_allrecipe_response(url):
    """
    This function accepts a Universal Resource Locator (url), sends a GET request, receives the response, and parses the response using BeautifulSoup. The output is a pandas DataFrame with 1 row representing data extracted from the URL.
    """

    # Define default options for Selenium webdriver
    chrome_options = webdriver.ChromeOptions()

    # Initiate webdriver, with command executor found within the Selenium Grid docker container
    # For this line to work, the `selenium/standalone-chrome:118.0` docker image
    driver = webdriver.Remote(
        command_executor = "http://172.19.0.2:4444",
        options = chrome_options
    )
    print(f"\t Initiated driver")
    
    # Use the driver to send a GET request
    driver.get(url)
    print(f"\t Sent GET request and received response")

    # Parse the response text using Beautiful Soup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    print(f"\t Soupified response")
    
    # Quit the driver
    driver.quit()
    print(f"\t Ended driver")

    ### Data Extraction from response ###
    # Initiate a blank dictionary to store values
    temp_dict = dict()
    print(f"\t Initiated blank dictionary")
    
    # column 00: url, the url of the recipe
    try:
        temp_dict.update({"recipe_url": url})
    except:
        temp_dict.update({"recipe_url": np.NaN})
    
    # column 01: title, the title of the recipe
    try:
        temp_dict.update({"title":
                          soup.find("h1", {"id": re.compile("^article-heading_*")}).get_text().strip(' \t\n\r')
                         })
    except:
        temp_dict.update({"title": np.NaN})
        
    # column 02: image, any image urls found within the recipes
    try:
        t_main_img = [img.get("src") for img in soup.find("div", {"class": "loc article-content"}).find_all("img") if img.get("src") != ""]
        t_sub_img = [img.get("data-src") for img in soup.find("div", {"class": "loc article-content"}).find_all("img") if img.get("data-src") != None]
        t_img = list((set(t_main_img+t_sub_img)))
        temp_dict.update({"image":t_img})
    except:
        temp_dict.update({"image": np.NaN})
    
    # column 03: rating_average, the target feature
    try:
        temp_dict.update({"rating_average":
                          float(soup.find("div", {"id": re.compile("mntl-recipe-review-bar__rating_*")}).get_text().strip(' \t\n\r'))
                         })
    except:
        temp_dict.update({"rating_average": np.NaN})
        
    # column 04: rating_count, the number of ratings for the recipe
    try:
        temp_dict.update({"rating_count":
                          soup.find("div", {"id": re.compile("^mntl-recipe-review-bar__rating-count_*")}).get_text().strip(' \t\n\r()')
                         })
    except:
        temp_dict.update({"rating_count": np.NaN})
        
    # column 05: review_count, the number of reviews for the recipe
    try:
        temp_dict.update({"review_count":
                         soup.find("div", {"id": re.compile("^mntl-recipe-review-bar__comment-count_*")}).get_text().strip(' \t\n\r()')
                         })
    except:
        temp_dict.update({"review_count": np.NaN})
        
    # column 06: description, the description section beneath each title of the recipe
    try:
        temp_dict.update({"description":
                         soup.find("p", {"id" : re.compile("^article-subheading_*")}).get_text().strip(' \t\n\r')
                         })
    except:
        temp_dict.update({"description": np.NaN})
        
    # column 07: update_date, the last date of update for the recipe
    try:
        temp_dict.update({"update_date":
                         soup.find_all("div", {"class": re.compile("^mntl-attribution__item-date*")})[0].get_text()
                         })
    except:
        temp_dict.update({"update_date": np.NaN})    
    
    # column 08: ingredient, a list of ingredients and their amounts
    try:
        temp_dict.update({"ingredient":
                         [li.get_text().strip(' \t\n\r') for li in soup.find("div", {"id": re.compile("^mntl-structured-ingredients_*")}).find_all("li")]
                         })
    except:
        temp_dict.update({"ingredient": np.NaN})
        
    # column 09: direction, a list of cooking directions or instructions
    try:
        temp_dict.update({"direction":
                          [li.get_text().strip(' \t\n\r') for li in soup.find("div", {"id": re.compile("^recipe__steps-content_*")}).find_all("li")]
                         })
    except:
        temp_dict.update({"direction": np.NaN})
        
    # column 10: nutrition_summary, a dictionary of nutritional information summary
    try:
        tag = soup.find("div", {"id": re.compile("^mntl-nutrition-facts-summary_*")})
        
        t_value = [line.get_text() for line in tag.find_all("td",{"class":"mntl-nutrition-facts-summary__table-cell type--dog-bold"})]
        header_1 = [line.get_text() for line in tag.find_all("td",{"class":"mntl-nutrition-facts-summary__table-cell type--dogg"})]
        header_2 = [line.get_text() for line in tag.find_all("td",{"class":"mntl-nutrition-facts-summary__table-cell type--dog"})]
        t_header = header_1+header_2
        
        temp_dict.update({"nutrition_summary":
                          {key:value for (key,value) in zip(t_header,t_value)}
                         })
    except:
        temp_dict.update({"nutrition_summary": np.NaN})
        
    # column 11: nutrition_detail, a dictionary of detailed nutritional information
    try:
        temp_dict.update({"nutrition_detail":
                          pd.read_html(str(soup.find_all("table",{"class": "mntl-nutrition-facts-label__table"})))[0]\
                          .iloc[:,0].to_list()
                         })
    except:
        temp_dict.update({"nutrition_detail": np.NaN})
    
    # column 12: time, a dictionary containing time related values in the recipe
    try:
        t_value = [div.get_text().strip(' \t\n\r') for div in soup.find("div", {"id": re.compile("^recipe-details_*")}).find_all("div", {"class":re.compile("^mntl-recipe-details__val*")})]
        t_header = [div.get_text().strip(' \t\n\r') for div in soup.find("div", {"id": re.compile("^recipe-details_*")}).find_all("div", {"class":re.compile("^mntl-recipe-details__la*")})]
        temp_dict.update({"time":
                          {key:value for (key,value) in zip(t_header,t_value)}
                         })
    except:
        temp_dict.update({"time": np.NaN})    
        
    # column 13: label, a list containing the labels or tags associated with the recipe
    try:
        temp_dict.update({"label":
                        [label.get_text() for label in soup.find("div", {"class":re.compile("^loc article-header")}).find_all("span",{"class":"link__wrapper"})]
                         })
    except:
        temp_dict.update({"label": np.NaN})    
        
    # column 14: review_dict, dictionary containing a JSON dictionary of reviews and other data elements of the webpage
    try:
        temp_dict.update({"review_dict":
                         ast.literal_eval(
                             soup.find('script',{"class":"comp allrecipes-schema mntl-schema-unified"}).text
                         )})
    except:
        temp_dict.update({"review_dict": np.NaN})
    
    # column 15: description_additional, additional description if available for the recipe
    try:
        temp_dict.update({"description_additional":
                         [p.get_text().strip(' \t\n\r') for p in soup.find_all('p',{"class":re.compile("^mntl-sc-block*")})]
                         })
    except:
        temp_dict.update({"description_additional": np.NaN})
        
    # Create a DataFrame with 1 row using the above data scraped into temp_dict
    temp_df = pd.DataFrame({k: pd.Series([v]) for k,v in temp_dict.items()}) 
    print(f"\t Converted dictionary to DataFrame.")
    
    return temp_df

# Accept URL input
input = sys.argv[1]
output = sys.argv[2]

# Run the function
print(f"\t Getting response")
raw_data_df = get_allrecipe_response(input)

# Store the output
print(f"\t Stored DataFrame into CSV")
raw_data_df.to_csv(output, header = True, index = False)

