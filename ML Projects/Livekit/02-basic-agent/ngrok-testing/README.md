# Query Processing API with ngrok

This API service processes user queries using OpenAI's GPT model and exposes the endpoint through ngrok for iOS app integration.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the same directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Run the server:
```bash
python ngrok.py
```

The server will start and display an ngrok URL that you can use to access the API from your iOS app.

## API Usage

### Endpoint: `/process_query`
- Method: POST
- Content-Type: application/json

Request body:
```json
{
    "query": "Your question here",
    "user_id": "optional_user_identifier"
}
```

Response:
```json
{
    "response": "AI generated response",
    "status": "success"
}
```

## iOS Integration

In your iOS app, you can make HTTP requests to the ngrok URL using URLSession or any networking library. Example:

```swift
let url = URL(string: "YOUR_NGROK_URL/process_query")!
var request = URLRequest(url: url)
request.httpMethod = "POST"
request.setValue("application/json", forHTTPHeaderField: "Content-Type")

let body = [
    "query": "Your question here",
    "user_id": "optional_user_id"
]

request.httpBody = try? JSONSerialization.data(withJSONObject: body)

URLSession.shared.dataTask(with: request) { data, response, error in
    // Handle response
}.resume()
```

## Notes

- The ngrok URL changes each time you restart the server
- In production, you should:
  - Replace the CORS `allow_origins=["*"]` with specific origins
  - Use proper authentication
  - Consider using a more permanent hosting solution
  - Adjust the OpenAI model parameters as needed