![alt text](minimum_length.png)

Currently supports symbolic and english formalizations

USAGE: (client not yet integrated)

python3 server/app.py

curl -X POST http://127.0.0.1:3000/formalize \
     -H "Content-Type: application/json" \
     -d '{"fileUrl":"./uploads/'attention.pdf'", "formatType":"logic"}'


You will need your own API key for now 😔 