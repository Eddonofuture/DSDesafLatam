import sys
import requests
import json

#url = "https://reqres.in/api/users"
#method= "GET"
#payload = ""

def request(method,  url , payload = ""):

    headers = {
        'User-Agent': "PostmanRuntime/7.11.0",
        'Accept': "*/*",
        'Cache-Control': "no-cache",
        'Postman-Token': "37bdc55f-fb3f-4355-9c13-08df245e0140,e6de8a3d-1ffd-4081-ba9e-eb38451ab092",
        'Host': "reqres.in",
        'accept-encoding': "gzip, deflate",
        'Connection': "keep-alive",
        'cache-control': "no-cache"
        }
    response = requests.request(method , url , data=payload, headers=headers)
    
        
    if(method == 'GET' or method == 'POST' or method == 'PUT'):
        if (method == 'GET'):
            resp = json.loads(response.text)
            for users in resp["data"]:
                print (users)
        elif (method == 'POST'):
            resp = json.loads(response.text)
            return ( resp)
        elif (method == 'PUT'):
            resp = json.loads(response.text)
            return ( resp)
        #return response.status_code, json.loads(response.text)
    else:
        #print (response.status_code)
        return(response)

#request(method ,url,payload)
