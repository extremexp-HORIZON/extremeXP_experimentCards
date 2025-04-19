from flask import Flask, jsonify, render_template, request, flash, redirect, url_for
from app.models import db, Experiment,ExperimentRequirement, ExperimentModel, EvaluationMetric, LessonLearnt, ExperimentDataset, ExperimentConstraint
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
from http.client import HTTPException

logging.basicConfig(level=logging.DEBUG)
from flask_swagger_ui import get_swaggerui_blueprint

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.yaml'  # Path to your Swagger YAML file
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL)


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
    return app, db
app, db = create_app()
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


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
        response = requests.put(metrics_url, headers=headers, data=json.dumps(payload))
        if response.status_code in [200, 201]:
            logging.info("Response JSON:", response.json()) 
            return {"success": True, "data": response.json()}
        
        # Handle unexpected status codes
        logging.info(f"Unexpected status code: {response.status_code} - {response.text}")
        return {"success": False, "error": f"Unexpected status code: {response.status_code}"}
    
    except requests.exceptions.RequestException as e:
        # Log the error and return a failure response
        logging.info(f"Error during API call: {e}")
        return {"success": False, "error": str(e)}

@app.route("/form_lessons_learnt/<experiment_id>", methods=["GET"])
def form(experiment_id):
    return render_template("form_lessons_learnt.html", experiment_id=experiment_id)


@app.route('/experiments', methods=['GET'])
def list_experiments():
    experiments = Experiment.query.all()
    return jsonify([{
        "experiment_id": e.experiment_id,
        "experiment_name": e.experiment_name,
        "lessons_learnt": [lesson.lessons_learnt for lesson in e.lessons],
        "experimentRatings": [lesson.experiment_rating for lesson in e.lessons],
    } for e in experiments])

@app.route("/message/<experiment_id>")
def message_page(experiment_id):
    status = request.args.get("status", "error")
    msg = request.args.get("msg", "Unknown error.")
    color = "#28a745" if status == "success" else "#dc3545"  # Green for success, red for error
    return render_template("message.html", msg=msg, color=color, experiment_id=experiment_id)

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/app/uploads') 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  
import logging

@app.route('/submit/<experiment_id>', methods=['POST'])
def submit_form(experiment_id):
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
                return redirect(url_for("message_page", experiment_id=experiment_id, status="error", msg="Invalid file format. Please upload a PDF file."))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pdf_file.save(file_path)

        # Validate form inputs
        try:
            experiment_rating = int(experiment_rating)
        except ValueError:
            return redirect(url_for("message_page", experiment_id=experiment_id, status="error", msg="Experiment rating must be an integer."))

        try:
            run_ratings = [int(r.strip()) for r in run_ratings.split(",")]
        except ValueError:
            return redirect(url_for("message_page", experiment_id=experiment_id, status="error", msg="Run ratings must be integers separated by commas."))

        if any(r < 1 or r > 7 for r in run_ratings):
            return redirect(url_for("message_page", experiment_id=experiment_id, status="error", msg="Each run rating must be between 1 and 7 (inclusive)."))

        # Add metrics to DAL
        response_lessons = add_metric(experiment_id, "lessonsLearnt", lessons_learnt, "string")
        response_experiment = add_metric(experiment_id, "experimentRating", str(experiment_rating), "string")
        response_runs = add_metric(experiment_id, "runRatings", str(run_ratings), "series")

         # Add metrics to DB
         # Add metrics to the database
        try:
            lesson_learnt_entry = LessonLearnt(
                lessons_learnt_id = f"lessons_learnt_{experiment_id}",
                experiment_id=experiment_id,
                lessons_learnt=lessons_learnt,
                experiment_rating=experiment_rating,
                run_rating=run_ratings
            )
            db.session.add(lesson_learnt_entry)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logging.error(f"Failed to insert metrics into the database: {e}")
            return redirect(url_for("message_page", experiment_id=experiment_id, status="error", msg=f"Failed to insert metrics into the database for experiment with id {experiment_id}."))


        if not response_lessons.get("success", False):
            return redirect(url_for("message_page", experiment_id=experiment_id, status="error", msg=f"Failed to insert lessons learnt metric for experiment with id {experiment_id}."))
        if not response_experiment.get("success", False):
            return redirect(url_for("message_page", experiment_id=experiment_id, status="error", msg=f"Failed to insert experiment rating metric for experiment with id {experiment_id}."))
        if not response_runs.get("success", False):
            return redirect(url_for("message_page", experiment_id=experiment_id, status="error", msg=f"Failed to insert run ratings metric for experiment with id {experiment_id}."))

        return redirect(url_for("message_page", experiment_id=experiment_id, status="success", msg=f"Metrics successfully inserted for experiment with id {experiment_id}!"))

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return redirect(url_for("message_page", experiment_id=experiment_id, status="error", msg="An unexpected error occurred. Please try again later."))

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

@app.route('/query_experiments_page', methods=['GET', 'POST'])
def query_example_new_sqlalchemy():
    results = []
    filters = {}

    if request.method == 'POST':
        # Get filter values from the form
        experiment_name = request.form.get('experiment_name')
        intent = request.form.get('intent')
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

def extract_experiment_data(workflows_list):
    experiment_info = {
        "train_model_implementations": [],
        "evaluation_model_implementation": None,
        "search_strategy_nn": "gridsearch",
        "search_strategy_rnn": "gridsearch",
        "dataset_path": None,
        "trained_model_output": None,
        "features": [],
        "parameters": {}
    }
    for workflow in workflows_list:
        tasks = workflow.get("tasks", [])
        workflow_id = workflow.get("workflowId", "unknown_workflow")
        for task in tasks:
            task_name = task.get("name", "")
            implementation_path = task.get("source_code", "")
            if "TrainModel" in task_name:
                experiment_info["train_model_implementations"].append(implementation_path)
                for dataset in task.get("input_datasets", []):
                    if dataset["name"] == "Features":
                        experiment_info["features"].append(dataset["name"])
                experiment_info["parameters"][workflow_id] = [{
                    "name": param.get("name"),
                    "type": param.get("type"),
                    "value": param.get("value")
                } for param in task.get("parameters", [])]
            if "EvaluateModel" in task_name:
                experiment_info["evaluation_model_implementation"] = implementation_path
            for dataset in task.get("input_datasets", []):
                if "ExternalDataFile" in dataset.get("name", ""):
                    experiment_info["dataset_path"] = dataset.get("uri")
            for dataset in task.get("output_datasets", []):
                if "TrainedModelFolder" in dataset.get("name", ""):
                    experiment_info["trained_model_output"] = dataset.get("uri")
    return experiment_info

def fetch_experiment_data(experiment_id):
    base_url = "https://api.expvis.smartarch.cz/api"
    headers = {"access-token": ACCESS_TOKEN}
    experiment_url = f"{base_url}/experiments/{experiment_id}"
    response = requests.get(experiment_url, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Experiment not found")
    data = response.json().get("experiment", {})
    workflow_ids = data.get("workflow_ids", [])
    workflows_list = []
    start_dates = []
    end_dates = []
    metrics_list = []
    metrics_semantic_types = []
    record_values = []

    for workflow_id in workflow_ids:
        workflow_endpoint = f"{base_url}/workflows/{workflow_id}"
        workflow_response = requests.get(workflow_endpoint, headers=headers)
        if workflow_response.status_code == 200:
            workflow_data = workflow_response.json().get("workflow", {})
            start_date = workflow_data.get("start")
            end_date = workflow_data.get("end")
            # Append valid dates or None if the date is missing or invalid
            start_dates.append(start_date if start_date else None)
            end_dates.append(end_date if end_date else None)
            status = workflow_data.get("status", "NA")
            metrics = workflow_data.get("metrics", [])
            workflows_list.append({
                "workflowId": workflow_id,
                "tasks": workflow_data.get("tasks", []),
                "status": status
            })
            for metric_entry in metrics:
                for metric_id, metric_details in metric_entry.items():
                    records = metric_details.get("records")
                    print(records)
                    if records is not None:
                        if isinstance(records, list):
                            record_values = [record.get("value") for record in records if isinstance(record, dict)]
                        else:
                            record_values = records.get("value") if isinstance(records, dict) else []
                    else:
                        record_values = metric_details.get("value")

                    metrics_list.append({
                        "metricId": metric_id,
                        "name": metric_details.get("name"),
                        "semantic_type": metric_details.get("semantic_type"),
                        "type": metric_details.get("type"),
                        "records": record_values,
                        "aggregation": metric_details.get("aggregation", {})
                    })
                    if "semantic_type" in metric_details:
                        metrics_semantic_types.append(metric_details["semantic_type"])
            user_interactions = [{"stoppedByUser": "NA"}] * len(record_values)
    extracted_data = extract_experiment_data(workflows_list)
    experiment_data = {
        "experimentCard": {
            "experimentInfo": {
                "experimentId": data.get("id"),
                "experimentName": data.get("name"),
                "intent": "Classification",
                "constraints": [{
                    "id": "ConstraintXX",
                    "on": "SVM",
                    "isHard": True,
                    "how": "Use"
                }],
                "requirements": [{
                    "id": "RequirementXX",
                    "metric": "Accuracy",
                    "method": "CrossValidation",
                    "objective": "Maximize"
                }],
                "experimentStartDate": start_dates,
                "experimentEndDate": end_dates,

                "status": status,
                "collaborators": [],  # talk to Ilias
                "userInteraction": user_interactions,
            },
            "variabilityPoints": {
                "dataSet": {
                    "id_dataset":extracted_data["dataset_path"],  # to be changed
                    "name": None,  # talk to Orestis
                    "zenoh_key_expr": None,  # talk to Orestis
                    "reviewer_score": None,
                    "path": extracted_data["dataset_path"]
                },
                "model": {
                    "algorithm": extracted_data["train_model_implementations"],
                    # "evaluationModel": extracted_data["evaluation_model_implementation"],
                    "parameters": extracted_data["parameters"],
                    "modelJSON": data.get("modelJSON")
                },
                "processing": {
                    "workflow": workflows_list,
                    "features": extracted_data["features"]
                }

            },
            "evaluation": {
                "runMetrics": metrics_list,
                "metrics": metrics_semantic_types
            }
        }
    }
    return experiment_data


def insert_data_from_json(data):
    try:

        # Insert Experiment data
        experiment_info = data["experimentCard"]["experimentInfo"]
        print(experiment_info)
        experiment = Experiment(
            experiment_id=experiment_info["experimentId"],
            experiment_name=experiment_info["experimentName"],
            experiment_start=experiment_info["experimentStartDate"][0],  # Assuming the first start date
            experiment_end=experiment_info["experimentEndDate"][-1],  # Assuming the last end date
            status=experiment_info["status"],
            collaborators=experiment_info["collaborators"],
            intent=experiment_info["intent"]
        )
        db.session.add(experiment)

        # Insert Constraints
        for constraint in experiment_info["constraints"]:
            experiment_constraint = ExperimentConstraint(
                id=constraint["id"],
                on_component=constraint["on"],
                is_hard=constraint["isHard"],
                how=constraint["how"],
                experiment_id=experiment_info["experimentId"]
            )
            db.session.add(experiment_constraint)

        # Insert Requirements
        for requirement in experiment_info["requirements"]:
            experiment_requirement = ExperimentRequirement(
                id=requirement["id"],
                metric=requirement["metric"],
                method=requirement["method"],
                objective=requirement["objective"],
                experiment_id=experiment_info["experimentId"]
            )
            db.session.add(experiment_requirement)

        # Insert Dataset
        dataset_info = data["experimentCard"]["variabilityPoints"]["dataSet"]
        experiment_dataset = ExperimentDataset(
            id_dataset=dataset_info["id_dataset"],
            name=dataset_info["name"],
            zenoh_key_expr=dataset_info["zenoh_key_expr"],
            reviewer_score=dataset_info["reviewer_score"],
            experiment_id=experiment_info["experimentId"]
        )
        db.session.add(experiment_dataset)

        # Insert Model
        model_info = data["experimentCard"]["variabilityPoints"]["model"]
        for workflow_id, parameters in model_info["parameters"].items():
            for param in parameters:
                experiment_model = ExperimentModel(
                    algorithm=model_info["algorithm"],
                    parameter=param["name"],
                    parameter_value=param["value"],
                    experiment_id=experiment_info["experimentId"]
                )
                db.session.add(experiment_model)

        # Insert Evaluation Metrics
        for metric in data["experimentCard"]["evaluation"]["runMetrics"]:
            evaluation_metric = EvaluationMetric(
                metric_id=metric["metricId"],
                name=metric["name"],
                value=metric["records"],
                # semantic_type=metric["semantic_type"],
                experiment_id=experiment_info["experimentId"]
            )
            db.session.add(evaluation_metric)

        db.session.commit()
        print("Data inserted successfully!")

    except Exception as e:
        db.session.rollback()
        print(f"Error inserting data: {e}")
    finally:
        db.session.close()

@app.route("/newExperiment/<experiment_id>", methods=["POST"])
def get_experiment(experiment_id):
    experiment_data = fetch_experiment_data(experiment_id)
    print(experiment_data)
    folder_path = os.path.join(os.getcwd(), "experiment_cards_json")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, f"{experiment_id}.json")
    try:
        with open(file_path, "w") as json_file:
            json.dump(experiment_data, json_file, indent=4)
            insert_data_from_json(experiment_data)
        return jsonify({"message": f"Experiment data saved successfully to {file_path}"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save experiment data: {str(e)}"}), 500


@app.route("/updateExperiment/<experiment_id>", methods=["POST"])
def update_experiment(experiment_id):
    experiment_data = fetch_experiment_data(experiment_id)

    if not experiment_data:
        return jsonify({"error": "Experiment not found"}), 404
    folder_path = os.path.join(os.getcwd(), "experiment_cards_json")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f"{experiment_id}.json")
    try:
        with open(file_path, "w") as json_file:
            json.dump(experiment_data, json_file, indent=4)
            insert_data_from_json(experiment_data)
        return jsonify({"message": f"Experiment data saved successfully to {file_path}"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save experiment data: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)