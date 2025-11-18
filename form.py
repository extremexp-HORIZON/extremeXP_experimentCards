import json
import logging
import os
from itertools import groupby
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, redirect, url_for
from flask_swagger_ui import get_swaggerui_blueprint
from sqlalchemy.sql import func
from werkzeug.exceptions import NotFound
from werkzeug.utils import secure_filename

from app.config import Config
from app.ingest import load_and_insert
from app.models import (
    EvaluationMetric,
    Experiment,
    ExperimentConstraint,
    ExperimentDataset,
    ExperimentModel,
    ExperimentRequirement,
    LessonLearnt,
    db,
)


load_dotenv()

logging.basicConfig(level=logging.INFO)

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.yaml'  
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL)


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)

    with app.app_context():
        db.create_all()
        try:
            load_and_insert(db)
            app.logger.info("Database populated successfully!")
        except Exception as e:
            app.logger.exception("Error populating database: %s", e)
    return app, db
app, db = create_app()
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
if not ACCESS_TOKEN:
    raise RuntimeError("ACCESS_TOKEN environment variable is not set.")

BASE_URL = "https://api.dal.extremexp-icom.intracom-telecom.com/api"
EXCLUDED_METRICS = {"lessonsLearnt", "experimentRating", "runRatings"}


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

    try:
        response = requests.put(metrics_url, headers=headers, json=payload, timeout=10)
        if response.status_code in [200, 201]:
            logging.info("Metric updated via DAL API: %s", response.json())
            return {"success": True, "data": response.json()}

        logging.info("Unexpected status code %s: %s", response.status_code, response.text)
        return {"success": False, "error": f"Unexpected status code: {response.status_code}"}

    except requests.exceptions.RequestException as e:
        logging.info("Error during DAL API call: %s", e)
        return {"success": False, "error": str(e)}


def _query_experiment_entities(experiment_id: str):
    experiment = Experiment.query.get_or_404(experiment_id)
    requirements = ExperimentRequirement.query.filter_by(experiment_id=experiment_id).all()
    models = ExperimentModel.query.filter_by(experiment_id=experiment_id).all()
    datasets = ExperimentDataset.query.filter_by(experiment_id=experiment_id).all()
    lessons = LessonLearnt.query.filter_by(experiment_id=experiment_id).all()
    evaluation = EvaluationMetric.query.filter_by(experiment_id=experiment_id).all()
    return experiment, requirements, models, datasets, lessons, evaluation


def _log_experiment_snapshot(experiment, requirements, models, datasets, lessons, evaluation):
    logging.info("Experiment: %s", experiment)
    logging.info("Requirements: %s", requirements)
    logging.info("Models: %s", models)
    logging.info("Datasets: %s", datasets)
    logging.info("Lessons: %s", lessons)
    for metric in evaluation:
        logging.info("Evaluation metric: %s", vars(metric))


def _filter_metrics(metrics: Iterable[EvaluationMetric], excluded: Sequence[str]):
    excluded_set = set(excluded)
    return [metric for metric in metrics if metric.name not in excluded_set]


def _save_pdf_file(pdf_file):
    if not pdf_file or not pdf_file.filename:
        return None
    filename = secure_filename(pdf_file.filename)
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Invalid file format. Please upload a PDF file.")
    upload_folder = app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, filename)
    pdf_file.save(file_path)
    return file_path


def _parse_run_ratings(raw_ratings: Optional[str]) -> List[int]:
    if not raw_ratings:
        raise ValueError("Run ratings are required.")
    normalized = raw_ratings.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    tokens = [token.strip() for token in normalized.split(",")]
    if not tokens or any(token == "" for token in tokens):
        raise ValueError("Run ratings must be integers separated by commas.")
    try:
        ratings = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("Run ratings must be integers separated by commas.") from exc
    if not ratings:
        raise ValueError("Run ratings are required.")
    if any(r < 1 or r > 7 for r in ratings):
        raise ValueError("Each run rating must be between 1 and 7 (inclusive).")
    return ratings


def _upsert_metric(experiment_id: str, metric_name: str, metric_value: str):
    metric_id = f"{experiment_id}_{metric_name}"
    metric_row = EvaluationMetric.query.filter_by(metric_id=metric_id, experiment_id=experiment_id).first()
    if metric_row:
        metric_row.name = metric_name
        metric_row.value = metric_value
    else:
        db.session.add(
            EvaluationMetric(
                metric_id=metric_id,
                experiment_id=experiment_id,
                name=metric_name,
                value=metric_value,
            )
        )


def _is_ajax_request() -> bool:
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


def _submit_response(success: bool, message: str, experiment_id: str, http_status: int = 200):
    if _is_ajax_request():
        payload = {"status": "success" if success else "error", "message": message}
        return jsonify(payload), http_status
    status_param = "success" if success else "error"
    return redirect(
        url_for(
            "message_page",
            experiment_id=experiment_id,
            status=status_param,
            msg=message,
        )
    )
    
@app.after_request
def add_header(response):
    # Allow embedding in iframes
    response.headers['X-Frame-Options'] = 'ALLOWALL'
    return response
@app.route('/')
def embed_test():
    return render_template('embed_test.html')
@app.route('/experiment_details_realData/<experiment_id>', methods=['GET'])
def experiment_details_realData(experiment_id):
    experiment, requirements, models, datasets, lessons, evaluation = _query_experiment_entities(experiment_id)
    _log_experiment_snapshot(experiment, requirements, models, datasets, lessons, evaluation)

    variability_points = {"dataSet": datasets, "model": models}
    filtered_evaluation = _filter_metrics(evaluation, EXCLUDED_METRICS)

    logging.info("Variability Points: %s", variability_points)

    return render_template(
        'experiment_details_realData.html',
        experiment=experiment,
        requirements=requirements,
        models=models,
        datasets=datasets,
        lessons=lessons,
        variabilityPoints=variability_points,
        evaluation=filtered_evaluation
    )

@app.route('/experiment_details/<experiment_id>', methods=['GET'])
def experiment_details(experiment_id):
    experiment, requirements, models, datasets, lessons, evaluation = _query_experiment_entities(experiment_id)
    _log_experiment_snapshot(experiment, requirements, models, datasets, lessons, evaluation)

    variabilityPoints = {
        "dataSet": {
            "name": "Example Dataset",
            "zenoh_key_expr": "example_key",
            "reviewer_score": 85
        },
        "model": {
            "algorithm": ["Algorithm1", "Algorithm2"],
            "parameters": ["Param1", "Param2"]
        },
        "processing": {
            "workflow": [
                {
                    "workflowId": "Workflow1",
                    "tasks": [{"name": "Task1"}, {"name": "Task2"}]
                }
            ]
        }
    }
    evaluation = {
        "metrics": ["Metric1", "Metric2"],
        "runMetrics": ["RunMetric1", "RunMetric2"]
    }

    return render_template(
        'experiment_details.html',
        experiment=experiment,
        requirements=requirements,
        models=models,
        datasets=datasets,
        lessons=lessons,
        variabilityPoints=variabilityPoints,
        evaluation=evaluation
    )

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

@app.route('/submit/<experiment_id>', methods=['POST'])
def submit_form(experiment_id):
    try:
        lessons_learnt = request.form.get('lessonsLearnt', '').strip()
        experiment_rating_raw = request.form.get('experimentRating')
        run_ratings_raw = request.form.get('runRatings')
        pdf_file = request.files.get('pdfFile')

        logging.info("Received submission for %s", experiment_id)
        logging.debug("Lessons learnt: %s", lessons_learnt)
        logging.debug("Experiment rating: %s", experiment_rating_raw)
        logging.debug("Run ratings: %s", run_ratings_raw)

        try:
            _save_pdf_file(pdf_file)
        except ValueError as exc:
            return _submit_response(False, str(exc), experiment_id, http_status=400)

        try:
            experiment_rating = int(experiment_rating_raw)
        except (TypeError, ValueError):
            return _submit_response(False, "Experiment rating must be an integer.", experiment_id, 400)

        try:
            run_ratings = _parse_run_ratings(run_ratings_raw)
        except ValueError as exc:
            return _submit_response(False, str(exc), experiment_id, 400)

        metric_payloads = [
            ("lessonsLearnt", lessons_learnt, "string"),
            ("experimentRating", str(experiment_rating), "string"),
            ("runRatings", json.dumps(run_ratings), "series"),
        ]

        for name, value, metric_type in metric_payloads:
            response = add_metric(experiment_id, name, value, metric_type)
            if not response.get("success"):
                return _submit_response(
                    False,
                    f"Failed to insert {name} metric for experiment with id {experiment_id}.",
                    experiment_id,
                    502,
                )

        try:
            entry = LessonLearnt.query.filter_by(
                lessons_learnt_id=f"lessons_learnt_{experiment_id}",
                experiment_id=experiment_id,
            ).first()
            if entry:
                entry.lessons_learnt = lessons_learnt
                entry.experiment_rating = experiment_rating
                entry.run_rating = run_ratings
            else:
                db.session.add(
                    LessonLearnt(
                        lessons_learnt_id=f"lessons_learnt_{experiment_id}",
                        experiment_id=experiment_id,
                        lessons_learnt=lessons_learnt,
                        experiment_rating=experiment_rating,
                        run_rating=run_ratings,
                    )
                )

            _upsert_metric(experiment_id, "lessonsLearnt", lessons_learnt)
            _upsert_metric(experiment_id, "experimentRating", str(experiment_rating))
            _upsert_metric(experiment_id, "runRatings", json.dumps(run_ratings))

            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logging.exception("Failed to insert metrics into the database: %s", exc)
            return _submit_response(
                False,
                f"Failed to insert metrics into the database for experiment with id {experiment_id}.",
                experiment_id,
                500,
            )

        return _submit_response(
            True,
            f"Metrics successfully inserted for experiment with id {experiment_id}!",
            experiment_id,
            200,
        )

    except Exception as exc:
        logging.exception("Unexpected error submitting form: %s", exc)
        return _submit_response(
            False,
            "An unexpected error occurred. Please try again later.",
            experiment_id,
            500,
        )

@app.route('/query_experiments_page', methods=['GET', 'POST'])
def query_experiments_page():
    filters = {}
    if request.method == 'POST':
        per_page = int(request.form.get('per_page', 10))
        filters = {
            'experiment_name': request.form.get('experiment_name', '').strip(),
            'intent': request.form.get('intent', '').strip(),
            'start_date': request.form.get('start_date', '').strip(),
            'end_date': request.form.get('end_date', '').strip(),
            'algorithm': request.form.get('algorithm', '').strip(),
            'metric_name': request.form.get('metric_name', '').strip(),
            'experiment_id': request.form.get('experiment_id', '').strip(),
        }
        query_args = {k: v for k, v in filters.items() if v}
        query_args['per_page'] = per_page
        query_args['page'] = 1
        return redirect(url_for('query_experiments_page', **query_args))

    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    filters = {
        'experiment_id': request.args.get('experiment_id', ''),
        'experiment_name': request.args.get('experiment_name', ''),
        'intent': request.args.get('intent', ''),
        'start_date': request.args.get('start_date', ''),
        'end_date': request.args.get('end_date', ''),
        'algorithm': request.args.get('algorithm', ''),
        'metric_name': request.args.get('metric_name', '')
    }

    experiment_query = Experiment.query

    if filters['experiment_id']:
        experiment_query = experiment_query.filter(Experiment.experiment_id == filters['experiment_id'])
    if filters['experiment_name']:
        experiment_query = experiment_query.filter(Experiment.experiment_name.ilike(f"%{filters['experiment_name']}%"))
    if filters['intent']:
        experiment_query = experiment_query.filter(func.lower(Experiment.intent) == filters['intent'].lower())
    if filters['start_date']:
        experiment_query = experiment_query.filter(func.date(Experiment.experiment_start) == filters['start_date'])
    if filters['end_date']:
        experiment_query = experiment_query.filter(func.date(Experiment.experiment_end) == filters['end_date'])
    if filters['algorithm']:
        experiment_query = experiment_query.join(ExperimentModel).filter(ExperimentModel.algorithm.ilike(f"%{filters['algorithm']}%"))
    if filters['metric_name']:
        experiment_query = experiment_query.join(EvaluationMetric).filter(EvaluationMetric.name.ilike(f"%{filters['metric_name']}%"))

    experiment_query = experiment_query.order_by(Experiment.experiment_start.desc()).distinct()
    pagination = experiment_query.paginate(page=page, per_page=per_page, error_out=False)

    grouped_results = []
    if pagination.items:
        experiment_ids = [exp.experiment_id for exp in pagination.items]
        details_query = db.session.query(
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
        ).filter(Experiment.experiment_id.in_(experiment_ids)).order_by(Experiment.experiment_id)

        results = details_query.all()
        def row_to_namespace(row):
            if hasattr(row, "_mapping"):
                data = dict(row._mapping)
            elif isinstance(row, dict):
                data = dict(row)
            else:
                data = row.__dict__.copy()
            return data

        for key, group in groupby(results, key=lambda x: x.experiment_id):
            rows = list(group)
            filtered_rows = []
            for row in rows:
                data = row_to_namespace(row)
                metric_name = data.get("metric_name")
                if metric_name and metric_name in EXCLUDED_METRICS:
                    continue
                filtered_rows.append(SimpleNamespace(**data))
            if not filtered_rows and rows:
                data = row_to_namespace(rows[0])
                data["metric_name"] = None
                data["metric_value"] = None
                filtered_rows.append(SimpleNamespace(**data))
            grouped_results.append(filtered_rows)

    filter_params = {k: v for k, v in filters.items() if v}

    def build_pagination_url(page_number):
        args = dict(filter_params)
        args['page'] = page_number
        args['per_page'] = per_page
        return url_for('query_experiments_page', **args)

    return render_template(
        'form_example_sqlalchemy.html',
        results=grouped_results,
        filters=filters,
        pagination=pagination,
        per_page=per_page,
        filter_params=filter_params,
        build_pagination_url=build_pagination_url
    )

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
    headers = {"access-token": ACCESS_TOKEN}
    experiment_url = f"{BASE_URL}/experiments/{experiment_id}"
    response = requests.get(experiment_url, headers=headers)
    if response.status_code != 200:
        raise NotFound("Experiment not found")
    data = response.json().get("experiment", {})
    workflow_ids = data.get("workflow_ids", [])
    workflows_list = []
    start_dates = []
    end_dates = []
    metrics_list = []
    metrics_semantic_types = []
    record_values = []

    for workflow_id in workflow_ids:
        workflow_endpoint = f"{BASE_URL}/workflows/{workflow_id}"
        workflow_response = requests.get(workflow_endpoint, headers=headers)
        if workflow_response.status_code == 200:
            workflow_data = workflow_response.json().get("workflow", {})
            start_date = workflow_data.get("start")
            end_date = workflow_data.get("end")

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
                    logging.debug("Metric %s raw records: %s", metric_id, records)
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
    experiment_info = data.get("experimentCard", {}).get("experimentInfo")
    experiment_id = experiment_info.get("experimentId") if experiment_info else "unknown"
    try:
        if not experiment_info:
            raise ValueError("experimentInfo section missing from payload")

        logging.info("Persisting experiment %s from JSON payload", experiment_id)
        experiment = Experiment(
            experiment_id=experiment_id,
            experiment_name=experiment_info["experimentName"],
            experiment_start=experiment_info["experimentStartDate"][0],  
            experiment_end=experiment_info["experimentEndDate"][-1], 
            status=experiment_info["status"],
            collaborators=experiment_info["collaborators"],
            intent=experiment_info["intent"]
        )
        db.session.add(experiment)

        for constraint in experiment_info["constraints"]:
            experiment_constraint = ExperimentConstraint(
                id=constraint["id"],
                on_component=constraint["on"],
                is_hard=constraint["isHard"],
                how=constraint["how"],
                experiment_id=experiment_id,
            )
            db.session.add(experiment_constraint)

        for requirement in experiment_info["requirements"]:
            experiment_requirement = ExperimentRequirement(
                id=requirement["id"],
                metric=requirement["metric"],
                method=requirement["method"],
                objective=requirement["objective"],
                experiment_id=experiment_id,
            )
            db.session.add(experiment_requirement)

        dataset_info = data["experimentCard"]["variabilityPoints"].get("dataSet")
        if dataset_info:
            experiment_dataset = ExperimentDataset(
                id_dataset=dataset_info.get("id_dataset"),
                name=dataset_info.get("name"),
                zenoh_key_expr=dataset_info.get("zenoh_key_expr"),
                reviewer_score=dataset_info.get("reviewer_score"),
                experiment_id=experiment_id,
            )
            db.session.add(experiment_dataset)

        model_info = data["experimentCard"]["variabilityPoints"].get("model") or {}
        for workflow_id, parameters in (model_info.get("parameters") or {}).items():
            for param in parameters:
                experiment_model = ExperimentModel(
                    algorithm=model_info.get("algorithm"),
                    parameter=param.get("name"),
                    parameter_value=param.get("value"),
                    experiment_id=experiment_id,
                )
                db.session.add(experiment_model)

        for metric in data["experimentCard"]["evaluation"]["runMetrics"]:
            evaluation_metric = EvaluationMetric(
                metric_id=metric["metricId"],
                name=metric["name"],
                value=metric["records"],
                experiment_id=experiment_id,
            )
            db.session.add(evaluation_metric)

        db.session.commit()
        logging.info("Experiment %s inserted successfully!", experiment_id)

    except Exception as e:
        db.session.rollback()
        logging.exception("Error inserting experiment %s: %s", experiment_id, e)
    finally:
        db.session.close()

@app.route("/newExperiment/<experiment_id>", methods=["POST"])
def get_experiment(experiment_id):
    experiment_data = fetch_experiment_data(experiment_id)
    logging.info("Fetched experiment %s from DAL API", experiment_id)
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
