import requests

resp = requests.post('http://test.suntechdrive.com/api/login', data={'username':'supowernet@gmail.com', 'password':'stanfords3l'})
print(resp)
