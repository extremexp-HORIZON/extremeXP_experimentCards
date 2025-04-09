from flask import Flask, jsonify, render_template, request, flash, redirect, url_for
from app.models import db, Experiment,ExperimentRequirement, ExperimentModel, EvaluationMetric, LessonLearnt, ExperimentDataset
from app.config import Config
from app.ingest import load_and_insert
from itertools import groupby
from operator import itemgetter
from sqlalchemy.sql import func
import requests
import os
import json
from werkzeug.utils import secure_filename
import logging

logging.basicConfig(level=logging.DEBUG)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)

    with app.app_context():
        db.create_all()
        try:
            load_and_insert(db)
            print("Database populated successfully!")
        except Exception as e:
            print(f"Error populating database: {str(e)}")
    return app
app = create_app()


ACCESS_TOKEN = "af880f22386d22f93e67a890bab7ebf2613b60e6"
EXPERIMENT_ID = "jZa-mJQBZTyxy1ACX1W8"
BASE_URL = "https://api.expvis.smartarch.cz/api"


# def add_metric(experiment_id, name, value, metric_type="string", kind="scalar", parent_type="experiment"):
#     headers = {"access-token": ACCESS_TOKEN, "Content-Type": "application/json"}
#     metrics_url = f"{BASE_URL}/metrics"
#     payload = {
#         "name": name,
#         "type": metric_type,
#         "kind": kind,
#         "value": value,
#         "parent_type": parent_type,
#         "parent_id": experiment_id
#     }
#     print(payload)
#     response = requests.put(metrics_url, headers=headers, data=json.dumps(payload))
#     print(response.json())
#     return response.json()

def add_metric(experiment_id, name, value, metric_type="string", kind="scalar", parent_type="experiment"):
    """
    Sends a PUT request to the API to add a metric.

    Args:
        experiment_id (str): The ID of the experiment.
        name (str): The name of the metric.
        value (str): The value of the metric.
        metric_type (str): The type of the metric (default: "string").
        kind (str): The kind of the metric (default: "scalar").
        parent_type (str): The parent type of the metric (default: "experiment").

    Returns:
        dict: The JSON response from the API.
    """
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

    try:
        # Send the PUT request
        response = requests.put(metrics_url, headers=headers, data=json.dumps(payload))
        if response.status_code in [200, 201]:
            logging.info("Response JSON:", response.json())  # Debugging
            return {"success": True, "data": response.json()}
        
        # Handle unexpected status codes
        logging.info(f"Unexpected status code: {response.status_code} - {response.text}")
        return {"success": False, "error": f"Unexpected status code: {response.status_code}"}
    
    except requests.exceptions.RequestException as e:
        # Log the error and return a failure response
        logging.info(f"Error during API call: {e}")
        return {"success": False, "error": str(e)}

@app.route("/form_lessons_learnt", methods=["GET"])
def form():
    return render_template("form_lessons_learnt.html")


@app.route('/experiments', methods=['GET'])
def list_experiments():
    experiments = Experiment.query.all()
    return jsonify([{
        "experiment_id": e.experiment_id,
        "experiment_name": e.experiment_name,
        "lessons_learnt": [lesson.lessons_learnt for lesson in e.lessons],
        "experimentRatings": [lesson.experiment_rating for lesson in e.lessons],
    } for e in experiments])

@app.route("/message")
def message_page():
    status = request.args.get("status", "error")
    msg = request.args.get("msg", "Unknown error.")
    color = "#28a745" if status == "success" else "#dc3545"  # Green for success, red for error
    return render_template("message.html", msg=msg, color=color)

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/app/uploads') 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  
import logging

@app.route('/submit', methods=['POST'])
def submit_form():
    try:
        lessons_learnt = request.form.get('lessonsLearnt')
        experiment_rating = request.form.get('experimentRating')
        run_ratings = request.form.get('runRatings')
        pdf_file = request.files.get('pdfFile')
        logging.info(f"Lessons Learnt: {lessons_learnt}")
        logging.info(f"Experiment Rating: {experiment_rating}")
        logging.info(f"Run Ratings: {run_ratings}")
        # Validate and save the uploaded file
        if pdf_file and pdf_file.filename:
            filename = secure_filename(pdf_file.filename)
            if not filename.endswith('.pdf'):
                return redirect(url_for("message_page", status="error", msg="Invalid file format. Please upload a PDF file."))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pdf_file.save(file_path)
        # else:
        #     return redirect(url_for("message_page", status="error", msg="No file uploaded."))

        # Validate form inputs
        try:
            experiment_rating = int(experiment_rating)
            logging.info(experiment_rating)
        except ValueError:
            return redirect(url_for("message_page", status="error", msg="Experiment rating must be an integer."))

        try:
            run_ratings = [int(r.strip()) for r in run_ratings.split(",")]
        except ValueError:
            return redirect(url_for("message_page", status="error", msg="Run ratings must be integers separated by commas."))

        if any(r < 1 or r > 7 for r in run_ratings):
            return redirect(url_for("message_page", status="error", msg="Each run rating must be between 1 and 7 (inclusive)."))

        # Add metrics to DAL
        logging.info(lessons_learnt)
        response_lessons = add_metric(EXPERIMENT_ID, "lessonsLearnt", lessons_learnt, "string")
        response_experiment = add_metric(EXPERIMENT_ID, "experimentRating", str(experiment_rating), "string")
        response_runs = add_metric(EXPERIMENT_ID, "runRatings", str(run_ratings), "series")
        logging.info(response_lessons)
        if not response_lessons.get("success", False):
            return redirect(url_for("message_page", status="error", msg="Failed to insert lessons learnt metric."))
        if not response_experiment.get("success", False):
            return redirect(url_for("message_page", status="error", msg="Failed to insert experiment rating metric."))
        if not response_runs.get("success", False):
            return redirect(url_for("message_page", status="error", msg="Failed to insert run ratings metric."))

        return redirect(url_for("message_page", status="success", msg="Metrics successfully inserted!"))

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return redirect(url_for("message_page", status="error", msg="An unexpected error occurred. Please try again later."))

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
            ei.intent,  
            m.algorithm, 
            em.name AS metric_name, 
            em.value AS metric_value,
            ll.lessons_learnt AS lessons_learnt,
            ll.experimentRating AS experimentRating
        FROM experiment_info ei
        LEFT JOIN experiment e ON e.experiment_info_id = ei.experiment_info_id
        LEFT JOIN variability_points vp ON vp.variability_points_id = ANY(e.variability_points_id)
        LEFT JOIN model m ON m.model_id = vp.model_id
        LEFT JOIN evaluation_metrics em ON em.metric_id = ANY(e.evaluation_metrics_id)
        LEFT JOIN lessons_learnt ll ON ll.lessons_learnt_id = e.lessons_learnt_id
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

@app.route('/query_example_new_sqlalchemy', methods=['GET', 'POST'])
def query_example_new_sqlalchemy():
    results = []
    filters = {}

    if request.method == 'POST':
        # Get filter values from the form
        experiment_name = request.form.get('experiment_name')
        intent = request.form.get('intent')
        print("Intent Filter Value:", intent)  # Debugging: Check the value of intent
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        algorithm = request.form.get('algorithm')
        metric_name = request.form.get('metric_name')

        # Build the query dynamically using SQLAlchemy
        query = db.session.query(
            Experiment.experiment_id,
            Experiment.experiment_name,
            Experiment.experiment_start,
            Experiment.experiment_end,
            Experiment.collaborators,
            Experiment.status,
            Experiment.intent,
            ExperimentRequirement.metric,
            ExperimentModel.algorithm,
            EvaluationMetric.name.label("metric_name"),
            EvaluationMetric.value.label("metric_value"),
            LessonLearnt.lessons_learnt,
            LessonLearnt.experiment_rating, 
            ExperimentDataset.name.label("dataset_name")
        ).join(
            LessonLearnt, Experiment.experiment_id == LessonLearnt.experiment_id, isouter=True
        ).join(
            ExperimentModel, Experiment.experiment_id == ExperimentModel.experiment_id, isouter=True
        ).join(
            EvaluationMetric, Experiment.experiment_id == EvaluationMetric.experiment_id, isouter=True
        ).join(
            ExperimentRequirement, Experiment.experiment_id == ExperimentRequirement.experiment_id, isouter=True
        ).join(
            ExperimentDataset, Experiment.experiment_id == ExperimentDataset.experiment_id, isouter=True
        )

        # Apply filters dynamically
        if experiment_name:
            query = query.filter(Experiment.experiment_name.ilike(f"%{experiment_name}%"))
        if intent:
            query = query.filter(func.lower(Experiment.intent) == intent.lower())
        if start_date:
            query = query.filter(func.date(Experiment.experiment_start) == start_date)
        if end_date:
            query = query.filter(func.date(Experiment.experiment_end) == end_date)
        if algorithm:
            query = query.filter(ExperimentModel.algorithm.ilike(f"%{algorithm}%"))
        if metric_name:
            query = query.filter(EvaluationMetric.name.ilike(f"%{metric_name}%"))

        # Execute the query
        results = query.all()
    else:
        # Default behavior for GET: Fetch all experiments
        results = db.session.query(
            Experiment.experiment_id,
            Experiment.experiment_name,
            Experiment.experiment_start,
            Experiment.experiment_end,
            Experiment.collaborators,
            Experiment.status,
            Experiment.intent,
            ExperimentRequirement.metric,
            ExperimentModel.algorithm,
            EvaluationMetric.name.label("metric_name"),
            EvaluationMetric.value.label("metric_value"),
            LessonLearnt.lessons_learnt,
            LessonLearnt.experiment_rating, 
            ExperimentDataset.name.label("dataset_name")
        ).join(
            LessonLearnt, Experiment.experiment_id == LessonLearnt.experiment_id, isouter=True
        ).join(
            ExperimentModel, Experiment.experiment_id == ExperimentModel.experiment_id, isouter=True
        ).join(
            EvaluationMetric, Experiment.experiment_id == EvaluationMetric.experiment_id, isouter=True
        ).join(
            ExperimentRequirement, Experiment.experiment_id == ExperimentRequirement.experiment_id, isouter=True
        ).join(
            ExperimentDataset, Experiment.experiment_id == ExperimentDataset.experiment_id, isouter=True
        ).all()
        
    grouped_results = []
    for key, group in groupby(results, key=lambda x: x.experiment_id):
        grouped_results.append(list(group))

    return render_template('form_example_sqlalchemy.html', results=grouped_results, filters=filters)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)