from flask import Flask, request, jsonify, render_template
from model import get_ai_response
import time

app = Flask(__name__)

# Route 1: Serve the HTML website
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route 2: The API endpoint that the website talks to
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    user_message = data.get('message')
    model_choice = data.get('model') # 'llama', 'mistral', or 'granite'
    
    if not user_message or not model_choice:
        return jsonify({"error": "Missing message or model selection"}), 400
    
    system_prompt = "Analyze the customer message, determine the sentiment, and draft a professional response."
    start_time = time.time()
    
    try:
        # Call our Groq LangChain engine
        result = get_ai_response(model_choice, system_prompt, user_message)
        
        # Add execution time to the JSON
        result['duration'] = round(time.time() - start_time, 2)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the local web server
    app.run(debug=True, port=5000)