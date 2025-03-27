from flask import Flask, request, redirect, url_for, render_template
import requests
import json
import psycopg2

app = Flask(__name__)
ACCESS_TOKEN = "af880f22386d22f93e67a890bab7ebf2613b60e6"
EXPERIMENT_ID = "jZa-mJQBZTyxy1ACX1W8"
BASE_URL = "https://api.expvis.smartarch.cz/api"

def add_metric(experiment_id, name, value, metric_type="string", kind="scalar", parent_type="experiment"):
    headers = {"access-token": ACCESS_TOKEN, "Content-Type": "application/json"}
    metrics_url = f"{BASE_URL}/metrics"
    payload = {
        "name": name,
        "type": metric_type,
        "kind": kind,
        "value": value,
        "parent_type": parent_type,
        "parent_id": experiment_id
    }
    response = requests.put(metrics_url, headers=headers, data=json.dumps(payload))
    return response.json()

@app.route("/", methods=["GET"])
def form():
    return render_template("form.html")

@app.route("/message")
def message_page():
    status = request.args.get("status", "error")
    msg = request.args.get("msg", "Unknown error.")
    color = "#28a745" if status == "success" else "#dc3545"  # Green for success, red for error
    return render_template("message.html", msg=msg, color=color)

@app.route("/submit", methods=["POST"])
def submit_form():
    lessonsLearnt = request.form.get("lessonsLearnt")
    experimentRating = request.form.get("experimentRating")
    runRatings = request.form.get("runRatings")

    try:
        experimentRating = int(experimentRating)
        run_ratings = [int(r.strip()) for r in runRatings.split(",")]
    except ValueError:
        return redirect(url_for("message_page", status="error", msg="Invalid input."))

    if any(r < 1 or r > 7 for r in run_ratings):
        return redirect(url_for("message_page", status="error", msg="Each run rating must be between 1 and 7."))

    response_lessons = add_metric(EXPERIMENT_ID, "lessonsLearnt", lessonsLearnt, "string")
    response_experiment = add_metric(EXPERIMENT_ID, "experimentRating", str(experimentRating), "string")
    response_runs = add_metric(EXPERIMENT_ID, "runRatings", str(run_ratings), "series")

    if "error" in response_lessons or "error" in response_experiment or "error" in response_runs:
        return redirect(url_for("message_page", status="error", msg="Failed to insert metrics."))

    return redirect(url_for("message_page", status="success", msg="Metrics successfully inserted!"))

def get_db_connection():
    conn = psycopg2.connect(
        dbname="experiment_db",
        user="user",
        password="password",
        host="172.17.0.0",
        port="5433"
    )
    return conn

@app.route('/experiment_query', methods=['GET', 'POST'])
def index():
    filters = {}
    results = []

    if request.method == 'POST':
        # Get filter values from the form
        experiment_name = request.form.get('experiment_name')
        status = request.form.get('status')

        # Build the query dynamically based on filters
        query = "SELECT * FROM experiment_info WHERE 1=1"
        params = []

        if experiment_name:
            query += " AND experiment_name ILIKE %s"
            params.append(f"%{experiment_name}%")
        
        if status:
            query += " AND status = %s"
            params.append(status)

        # Execute the query
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        conn.close()

    return render_template('query_form.html', results=results, filters=filters)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)