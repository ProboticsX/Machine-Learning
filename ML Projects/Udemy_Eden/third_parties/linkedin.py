import os
from dotenv import load_dotenv
import requests

load_dotenv()

def scrape_linkedin_profile(linkedin_profile_url:str, mock:bool=False):
    """ scape information from LinkedIn profiles,
        Manually scrape the information from LinkedIn profiles
    """
    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/ProboticsX/c7eb6795b52cfcc37d00b9817334d6be/raw/86a5a20102549d728dd491ea46d01fd59bbc87a9/shubham-jain.json"
        response = requests.get(
            linkedin_profile_url,
            timeout=10,
        )
    else:
        api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
        header_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}
        response = requests.get(
            api_endpoint,
            params={"url": linkedin_profile_url},
            headers=header_dic,
            timeout=10,
        )
    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")
    return data


if __name__ == "__main__":
    print(
        scrape_linkedin_profile(
            linkedin_profile_url="https://www.linkedin.com/in/shubham-jain-b31375137/",
            mock=True
        )
    )