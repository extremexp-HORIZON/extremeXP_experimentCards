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
        host="postgres",
        port="5432"
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


@app.route('/query_example', methods=['GET', 'POST'])
def query_example():
    results = []
    query = """
        SELECT experiment_id, experiment_name, experiment_start_time, experiment_end_time, collaborators, intent
        FROM experiment_cards_example
        WHERE 1=1
    """
    params = []

    if request.method == 'POST':
        # Get filter values from the form
        experiment_name = request.form.get('experiment_name')
        intent = request.form.get('intent')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        # Add filters to the query
        if experiment_name:
            query += " AND experiment_name ILIKE %s"
            params.append(f"%{experiment_name}%")
        if intent:
            query += " AND intent = %s"
            params.append(intent)
        if start_date:
            query += " AND DATE(experiment_start_time) = %s"
            params.append(start_date)
        if end_date:
            query += " AND DATE(experiment_end_time) = %s"
            params.append(end_date)

    # Execute the query
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('form_example.html', results=results)

@app.route('/query_example_new', methods=['GET', 'POST'])
def query_example_new():
    results = []
    query = """
        SELECT 
            ei.experiment_info_id AS experiment_id, 
            ei.experiment_name, 
            ei.experiment_start_date AS experiment_start_time, 
            ei.experiment_end_date AS experiment_end_time, 
            ei.collaborators, 
            ei.intent,  -- Corrected: intent is now referenced from experiment_info (ei)
            m.algorithm, 
            em.name AS metric_name, 
            em.value AS metric_value
        FROM experiment_info ei
        LEFT JOIN experiment e ON e.experiment_info_id = ei.experiment_info_id
        LEFT JOIN variability_points vp ON vp.variability_points_id = ANY(e.variability_points_id)
        LEFT JOIN model m ON m.model_id = vp.model_id
        LEFT JOIN evaluation_metrics em ON em.metric_id = ANY(e.evaluation_metrics_id)
        WHERE 1=1
    """
    params = []

    if request.method == 'POST':
        # Get filter values from the form
        experiment_name = request.form.get('experiment_name')
        intent = request.form.get('intent')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        algorithm = request.form.get('algorithm')
        metric_name = request.form.get('metric_name')

        # Add filters to the query
        if experiment_name:
            query += " AND ei.experiment_name ILIKE %s"
            params.append(f"%{experiment_name}%")
        if intent:
            query += " AND ei.intent = %s"  
            params.append(intent)
        if start_date:
            query += " AND DATE(ei.experiment_start_date) = %s"
            params.append(start_date)
        if end_date:
            query += " AND DATE(ei.experiment_end_date) = %s"
            params.append(end_date)
        if algorithm:
            query += " AND m.algorithm ILIKE %s"
            params.append(f"%{algorithm}%")
        if metric_name:
            query += " AND em.name ILIKE %s"
            params.append(f"%{metric_name}%")

    # Execute the query
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('form_example_new.html', results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)