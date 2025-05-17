from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend to talk to backend

@app.route('/')
def index():
    return "server is running"
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').strip().lower()
    if user_input == 'hi':
        return jsonify({'response': 'hello'})
    else:
        return jsonify({'response': f'I dont understand: {user_input}'})

if __name__ == '__main__':
    app.run(debug=True)
