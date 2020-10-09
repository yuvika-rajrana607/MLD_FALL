import requests

url = 'http://127.0.0.1:5000/predict_api'
r = requests.post(url,json={'stage':5, 'age':80, 'family_history':1 ,previous_cancer':1, 'smoker':1, 'diff_tumor':15, diff_psa':6, 'tea':2, 'rd_thrpy':1, chm_thrpy':1, 'rad_rem':1})

print(r.json())