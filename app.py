from flask import Flask, render_template, request
from form import app as form_app

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/query', methods=['POST'])
def query():
    query_text = request.form.get('query')
    return f"You searched for: {query_text}"
app.register_blueprint(form_app)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
