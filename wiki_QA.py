import json
import os 
import urllib3
openApiURL = "http://aiopen.etri.re.kr:8000/WikiQA"
accessKey = os.getenv('WIKIQA_API_KEY')

question = "버트 모델이 뭐야??"
type_ = "hybridqa"
requestJson = {
    "argument":{
        "question": question,
        "type": type_
    }
}
http = urllib3.PoolManager()
response = http.request(
"POST",
openApiURL,
headers={"Content-Type": "application/json; charset=UTF-8","Authorization": accessKey},
body=json.dumps(requestJson)
)

print("[responseCode] " + str(response.status))
print("[responBody]")
print(str(response.data,"utf-8"))
