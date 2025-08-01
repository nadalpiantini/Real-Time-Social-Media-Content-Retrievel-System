# Import necessary libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs
import time
import json

VERIFICATION_TIME = 10
SCROLL_PAUSE_TIME = 3
MAX_SCROLLS = False


# Function to extract text from a container
def get_text(container, selector, attributes):
    try:
        element = container.find(selector, attributes)
        if element:
            return element.text.strip()
    except Exception as e:
        print(e)
    return ""


# Initialize WebDriver for Chrome
def fetch_posts(email, password, page):
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    browser = webdriver.Chrome(options=options)

    # Open LinkedIn login page
    browser.get("https://www.linkedin.com/login")

    # Full screen mode
    browser.maximize_window()

    # Enter login credentials and submit
    elementID = browser.find_element(By.ID, "username")
    elementID.send_keys(email)
    elementID = browser.find_element(By.ID, "password")
    elementID.send_keys(password)
    elementID.submit()
    time.sleep(VERIFICATION_TIME)
    
    # Navigate to the posts page of the company
    browser.get(page)

    # Set parameters for scrolling through the page
    last_height = browser.execute_script("return document.body.scrollHeight")
    scrolls = 0
    no_change_count = 0

    # Scroll through the page until no new content is loaded
    while True:
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = browser.execute_script("return document.body.scrollHeight")
        no_change_count = no_change_count + 1 if new_height == last_height else 0
        if no_change_count >= 3 or (MAX_SCROLLS and scrolls >= MAX_SCROLLS):
            break
        last_height = new_height
        scrolls += 1

    # Parse the page source with BeautifulSoup
    company_page = browser.page_source
    linkedin_soup = bs(company_page.encode("utf-8"), "html.parser")

    # Extract post containers from the HTML
    containers = [
        container
        for container in linkedin_soup.find_all(
            "div", {"class": "feed-shared-update-v2"}
        )
        if "activity" in container.get("data-urn", "")
    ]
    return containers


def make_post_data(containers, username):
    # Import Supabase client
    from utils.supabase_client import supabase_client
    
    # Define a data structure to hold all the post information
    post_data = {}
    post_data["Name"] = username
    post_data["Posts"] = {}

    # Main loop to process each container
    for index, container in enumerate(containers):
        post_id = f"Post_{index+1}"
        post_text = get_text(
            container, "div", {"class": "feed-shared-update-v2__description-wrapper"}
        )
        post_text = post_text.replace("hashtag", "")
        post_text = post_text.replace('"', "")
        post_text = post_text.replace("\n", "")
        if post_text != "": # if content is not empty
            post_data["Posts"][post_id] = {
                "text": post_text, 
                "post_owner": username, 
                "name": username, 
                "source": "Linkedin"
            }

    # Save to Supabase (with JSON fallback)
    success = supabase_client.save_scraped_data(username, post_data)
    
    if success:
        print(f"✅ Successfully saved {len(post_data['Posts'])} posts for {username}")
    else:
        print(f"⚠️ Warning: Could not save posts for {username}")
    
    return post_data
