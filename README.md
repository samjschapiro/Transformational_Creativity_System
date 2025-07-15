![alt text](minimum_length.png)

Currently supports symbolic and english formalizations

USAGE: (client not yet integrated)

python3 server/app.py

curl -X POST http://PORT_URL/formalize \
     -H "Content-Type: application/json" \
     -d '{"file":'YOUR_PDF', "formatType":"logic/english"}'


You will need your own API key for now 😔 