import http.client

def lambda_handler(event, context):
    conn = http.client.HTTPConnection("52.78.239.63", 5000)
    conn.request("GET", "/classify")
    response = conn.getresponse()
    data = response.read().decode()
    conn.close()
    print(data)
    return data
