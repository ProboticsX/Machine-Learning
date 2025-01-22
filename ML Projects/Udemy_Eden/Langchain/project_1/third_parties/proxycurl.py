import requests

api_key = 'S1Rx5wNU_Nfi61IUrW3sug'
headers = {'Authorization': 'Bearer ' + api_key}
api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
params = {
    'linkedin_profile_url': 'https://www.linkedin.com/in/shubham-jain-b31375137/'
}
response = requests.get(api_endpoint,
                        params=params,
                        headers=headers)

# print (response.json())
print(response._content)