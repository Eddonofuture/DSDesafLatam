# yX7QIwvpPSxUvhN8earEM6QmnZVJJWzZA28wI1dY
import sys 
import json
import requests

def request(url , apiKey):
    #url = "https://api.nasa.gov/mars-photos/api/v1/rovers/"

    querystring = {"api_key": apiKey}

    payload = ""  
    headers = {
        'User-Agent': "PostmanRuntime/7.11.0",
        'Accept': "*/*",
        'Cache-Control': "no-cache",
        'Postman-Token': "06d253ed-9670-4343-9d08-e71f6384ee44,9791f52a-de15-44b3-844b-0bd4ec876577",
        'Host': "api.nasa.gov",
        'accept-encoding': "gzip, deflate",
        'Connection': "keep-alive",
        'cache-control': "no-cache"
        }

    response = requests.request("GET", url, data=payload, headers=headers, params=querystring)
    return json.loads(response.text)

def build_web_page(dic):
    listado = list(dic["photos"])[0:10]
    
    html = ""
    html += "<html>\n<head>\n</head>\n<body>\n<ul>\n"
    for photo in listado:

        html += "<li>"
        html += "<img src=\"{}\">".format(photo["img_src"])
        html += "</li>\n"
    html += "</ul>\n</body>\n</html>"    
    with open("output.html","w") as f:
        f.write(html)

#build_web_page(request("https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos", "yX7QIwvpPSxUvhN8earEM6QmnZVJJWzZA28wI1dY"))