1. Create an 'uploads' directory in the root, move your pdf there
2. Start the server; 'python3 server/app.py'

3. Run the following in your terminal: 
curl -X POST http://PORT_URL/formalize \
     -H "Content-Type: application/json" \
     -d '{"file":'YOUR_PDF', "formatType":"logic/english"}'

4. Check server/conceptual_space.png for image of conceptual space, and server/ouput for output files, including the .json log and output pdf
