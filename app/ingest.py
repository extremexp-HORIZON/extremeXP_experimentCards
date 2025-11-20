import os
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import random

from .models import (
    Experiment,
    ExperimentConstraint,
    ExperimentRequirement,
    ExperimentDataset,
    ExperimentModel,
    EvaluationMetric,
    LessonLearnt,
)

BASE_DIR = os.path.dirname(__file__)
EXPERIMENTS_PATH = os.path.join(BASE_DIR, "..", "dal_files", "experiments.json")
WORKFLOWS_PATH = os.path.join(BASE_DIR, "..", "dal_files", "workflows.json")
METRICS_PATH = os.path.join(BASE_DIR, "..", "dal_files", "metrics.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "files")

def _parse_iso_datetime(value: str):
    """Convert ISO8601 strings (with optional trailing Z) to datetime objects."""
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None

def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _extract_metric_value(metric_payload: Dict[str, Any]):
    if not metric_payload:
        return None
    if metric_payload.get("value") is not None:
        return str(metric_payload["value"])
    records = metric_payload.get("records")
    if isinstance(records, list):
        values = [
            str(record.get("value"))
            for record in records
            if isinstance(record, dict) and record.get("value") is not None
        ]
        return ",".join(values) if values else None
    if isinstance(records, dict):
        value = records.get("value")
        return str(value) if value is not None else None
    return None

def _load_index(path: str, root_key: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    entries = data.get(root_key, [])
    index = {}
    for entry in entries:
        if isinstance(entry, dict) and entry:
            key, value = next(iter(entry.items()))
            index[key] = value
    return index

def _group_workflows_by_experiment(workflow_index: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {}
    for workflow in workflow_index.values():
        exp_id = workflow.get("experimentId")
        if exp_id:
            grouped.setdefault(exp_id, []).append(workflow)
    return grouped

def _get_workflows_for_experiment(payload: Dict[str, Any], workflow_index: Dict[str, Any], workflow_groups: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    workflows = []
    for workflow_id in payload.get("workflow_ids", []):
        workflow = workflow_index.get(workflow_id)
        if workflow:
            workflows.append(workflow)
    if not workflows:
        workflows.extend(workflow_groups.get(payload.get("id"), []))
    return workflows

def _collect_constraints_from_workflows(workflows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    constraints = []
    seen = set()
    for workflow in workflows:
        workflow_id = workflow.get("id") or workflow.get("name") or "workflow"
        for task in workflow.get("tasks", []) or []:
            task_id = task.get("id") or task.get("name") or "task"
            constraint_id = f"{workflow_id}_{task_id}_constraint"
            if constraint_id in seen:
                continue
            seen.add(constraint_id)
            constraints.append({
                "id": constraint_id,
                "on_component": task.get("name") or task_id,
                "is_hard": True,
                "how": task.get("comment") or workflow.get("comment") or "Derived from workflow definition",
            })
    return constraints

def _collect_requirements_from_workflows(workflows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    requirements = []
    seen = set()
    def _add_requirement(source_id, metric):
        if not isinstance(metric, dict):
            return
        metric_name = metric.get("name") or "metric"
        req_id = f"{source_id}_{metric_name}_requirement"
        if req_id in seen:
            return
        seen.add(req_id)
        requirements.append({
            "id": req_id,
            "metric": metric_name,
            "method": metric.get("type") or metric.get("kind") or "unspecified",
            "objective": metric.get("semantic_type") or "optimize",
        })
    for workflow in workflows:
        workflow_id = workflow.get("id") or workflow.get("name") or "workflow"
        for metric in workflow.get("metrics", []) or []:
            _add_requirement(workflow_id, metric)
        for task in workflow.get("tasks", []) or []:
            task_id = task.get("id") or task.get("name") or workflow_id
            for metric in task.get("metrics", []) or []:
                _add_requirement(task_id, metric)
    return requirements

def _pick_dataset_from_workflows(workflows: List[Dict[str, Any]]):
    for workflow in workflows:
        inputs = workflow.get("input_datasets") or []
        if inputs:
            return inputs[0]
        outputs = workflow.get("output_datasets") or []
        if outputs:
            return outputs[0]
        for task in workflow.get("tasks", []) or []:
            inputs = task.get("input_datasets") or []
            if inputs:
                return inputs[0]
            outputs = task.get("output_datasets") or []
            if outputs:
                return outputs[0]
    return None

def _build_dataset_from_source(experiment: Experiment, dataset_source: Dict[str, Any]):
    if not dataset_source:
        return None
    dataset_id = (
        dataset_source.get("id")
        or dataset_source.get("id_dataset")
        or dataset_source.get("name")
        or dataset_source.get("uri")
    )
    if not dataset_id:
        dataset_id = f"{experiment.experiment_id}_dataset"
    reviewer_score = dataset_source.get("reviewer_score")
    if reviewer_score is not None:
        try:
            reviewer_score = int(reviewer_score)
        except (TypeError, ValueError):
            reviewer_score = None
    return ExperimentDataset(
        id_dataset=dataset_id,
        name=dataset_source.get("name"),
        zenoh_key_expr=dataset_source.get("zenoh_key_expr") or dataset_source.get("uri"),
        reviewer_score=reviewer_score,
        experiment=experiment,
    )

def _collect_model_parameters(experiment: Experiment, workflows: List[Dict[str, Any]]) -> List[ExperimentModel]:
    entries = []
    def _append_entry(algorithm, parameter, value):
        if not parameter:
            return
        entries.append(
            ExperimentModel(
                algorithm=algorithm,
                parameter=parameter,
                parameter_value=_safe_float(value),
                experiment=experiment,
            )
        )
    for workflow in workflows:
        algorithm = workflow.get("name") or workflow.get("id")
        for param in workflow.get("parameters", []) or []:
            _append_entry(algorithm, param.get("name"), param.get("value"))
        for task in workflow.get("tasks", []) or []:
            task_algo = task.get("name") or algorithm
            for param in task.get("parameters", []) or []:
                _append_entry(task_algo, param.get("name"), param.get("value"))
    return entries

def _normalize_metric_entry(metric: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(metric, dict):
        return {}
    if len(metric) == 1:
        key, value = next(iter(metric.items()))
        if isinstance(value, dict):
            normalized = value.copy()
            normalized.setdefault("id", key)
            return normalized
    return metric


def _collect_workflow_metrics(workflows: List[Dict[str, Any]], experiment_id: str, existing_ids: set) -> List[EvaluationMetric]:
    rows = []
    def _append_metric(source_id, metric, index):
        normalized_metric = _normalize_metric_entry(metric)
        if not normalized_metric:
            return
        metric_id = normalized_metric.get("id") or f"{source_id}_{index}"
        original_id = metric_id
        suffix = index
        while metric_id in existing_ids:
            suffix += 1
            metric_id = f"{original_id}_{suffix}"
        existing_ids.add(metric_id)
        value = normalized_metric.get("value")
        if value is None:
            records = normalized_metric.get("records")
            if isinstance(records, list):
                values = [
                    record.get("value")
                    for record in records
                    if isinstance(record, dict) and record.get("value") is not None
                ]
                value = ",".join(str(v) for v in values) if values else None
            elif isinstance(records, dict):
                value = records.get("value")
        rows.append(
            EvaluationMetric(
                metric_id=metric_id,
                experiment_id=experiment_id,
                name=normalized_metric.get("name"),
                value=str(value) if value is not None else None,
            )
        )
    for workflow in workflows:
        workflow_id = workflow.get("id") or workflow.get("name") or "workflow"
        for idx, metric in enumerate(workflow.get("metrics", []) or [], start=1):
            _append_metric(workflow_id, metric, idx)
        for task in workflow.get("tasks", []) or []:
            task_id = task.get("id") or task.get("name") or workflow_id
            for idx, metric in enumerate(task.get("metrics", []) or [], start=1):
                _append_metric(task_id, metric, idx)
    return rows

def _infer_timings_from_workflows(workflows: List[Dict[str, Any]]):
    starts = []
    ends = []
    for workflow in workflows:
        if workflow.get("start"):
            starts.append(_parse_iso_datetime(workflow["start"]))
        if workflow.get("end"):
            ends.append(_parse_iso_datetime(workflow["end"]))
        for task in workflow.get("tasks", []) or []:
            if task.get("start"):
                starts.append(_parse_iso_datetime(task["start"]))
            if task.get("end"):
                ends.append(_parse_iso_datetime(task["end"]))
    start_time = min([dt for dt in starts if dt], default=None)
    end_time = max([dt for dt in ends if dt], default=None)
    return start_time, end_time

def _add_placeholder_entries(experiment: Experiment, dataset_added: bool, model_added: bool, metric_added: bool):
    if not experiment.experiment_id.startswith("exp_"):
        return
    if not dataset_added:
        db.session.add(
            ExperimentDataset(
                id_dataset=f"{experiment.experiment_id}_dataset",
                name="Placeholder Dataset",
                zenoh_key_expr=f"/data/{experiment.experiment_id}.csv",
                reviewer_score=75,
                experiment=experiment,
            )
        )
    if not model_added:
        db.session.add(
            ExperimentModel(
                algorithm="PlaceholderModel",
                parameter="learning_rate",
                parameter_value=0.01,
                experiment=experiment,
            )
        )
    if not metric_added:
        db.session.add(
            EvaluationMetric(
                metric_id=f"{experiment.experiment_id}_metric",
                experiment_id=experiment.experiment_id,
                name="accuracy",
                value="0.90",
            )
        )

def _map_experiment_payload(payload: Dict[str, Any], metrics_index: Dict[str, Any]) -> Tuple[Experiment, List[EvaluationMetric]]:
    creator = payload.get("creator") or {}
    collaborators = [creator["name"]] if creator.get("name") else []

    experiment = Experiment(
        experiment_id=payload["id"],
        experiment_name=payload.get("name"),
        experiment_start=_parse_iso_datetime(payload.get("start")),
        experiment_end=_parse_iso_datetime(payload.get("end")),
        status=payload.get("status"),
        collaborators=collaborators,
        intent=payload.get("intent"),
    )

    metrics = []
    for index, metric_id in enumerate(payload.get("metric_ids", []), start=1):
        metric_payload = metrics_index.get(metric_id, {})
        metrics.append(
            EvaluationMetric(
                metric_id=metric_id,
                experiment_id=experiment.experiment_id,
                name=metric_payload.get("name") or f"metric_{index}",
                value=_extract_metric_value(metric_payload),
            )
        )

    return experiment, metrics

# def _map_experiment_payload(payload: Dict[str, Any]) -> Tuple[Experiment, List[EvaluationMetric]]:
#     """Build ORM objects from a raw experiment payload."""
#     creator = payload.get("creator", {})
#     collaborators = []
#     creator_name = creator.get("name")
#     if creator_name:
#         collaborators.append(creator_name)

#     experiment = Experiment(
#         experiment_id=payload["id"],
#         experiment_name=payload.get("name"),
#         experiment_start=_parse_iso_datetime(payload.get("start")),
#         experiment_end=_parse_iso_datetime(payload.get("end")),
#         status=payload.get("status"),
#         collaborators=collaborators,
#         intent=payload.get("intent"),
#     )

#     metrics = []
#     for index, metric_id in enumerate(payload.get("metric_ids", []), start=1):
#         metrics.append(
#             EvaluationMetric(
#                 metric_id=metric_id,
#                 experiment_id=experiment.experiment_id,
#                 name=f"metric_{index}",
#                 value=metric_id,
#             )
#         )

#     return experiment, metrics


def _seed_default_lessons(db):
    for i in range(1, 21):
        exp_id = f"exp_{i}"
        lesson_id = f"lessons_learnt_{i}"
        if not Experiment.query.get(exp_id):
            continue
        exists = LessonLearnt.query.filter_by(
            lessons_learnt_id=lesson_id, experiment_id=exp_id
        ).first()
        if exists:
            continue

        lesson_text = f"Lessons Learnt {i} text"
        rating = random.randint(1, 7)
        run_ratings = [random.randint(1, 10) for _ in range(random.randint(1, 4))]

        db.session.add(
            LessonLearnt(
                lessons_learnt_id=lesson_id,
                lessons_learnt=lesson_text,
                experiment_rating=rating,
                run_rating=run_ratings,
                experiment_id=exp_id,
            )
        )


def load_and_insert(db):
    """Populate the database using dal_files/experiments.json."""
    if not os.path.exists(EXPERIMENTS_PATH):
        raise FileNotFoundError(f"Experiments file not found at {EXPERIMENTS_PATH}")

    workflow_index = _load_index(WORKFLOWS_PATH, "workflows")
    workflow_groups = _group_workflows_by_experiment(workflow_index)
    metrics_index = _load_index(METRICS_PATH, "metrics")

    with open(EXPERIMENTS_PATH, "r", encoding="utf-8") as f:
        experiments_data = json.load(f).get("experiments", [])

    legacy_path = os.path.join(os.path.dirname(EXPERIMENTS_PATH), "..", "files")
    if os.path.isdir(legacy_path):
        for filename in sorted(os.listdir(legacy_path)):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(legacy_path, filename), "r", encoding="utf-8") as f:
                card = json.load(f)
            card_data = card.get("experimentCard", {})
            info = card_data.get("experimentInfo", {})
            exp_id = info.get("experimentId")
            if exp_id:
                experiments_data.append({
                    exp_id: {
                        "id": exp_id,
                        "name": info.get("experimentName"),
                        "intent": card_data.get("intent"),
                        "status": info.get("status"),
                        "collaborators": info.get("collaborators", []),
                        "workflow_ids": info.get("workflow_ids", []),
                        "model": card_data.get("model"),
                        "metadata": card_data.get("metadata"),
                    }
                })

    for entry in experiments_data:
        if not isinstance(entry, dict):
            continue
        exp_id, payload = next(iter(entry.items()))
        if Experiment.query.get(exp_id):
            continue

        experiment, metrics = _map_experiment_payload(payload, metrics_index)
        db.session.add(experiment)

        workflows = _get_workflows_for_experiment(payload, workflow_index, workflow_groups)
        start_time, end_time = _infer_timings_from_workflows(workflows)
        if experiment.experiment_start is None and start_time:
            experiment.experiment_start = start_time
        if experiment.experiment_end is None and end_time:
            experiment.experiment_end = end_time

        dataset_added = False
        model_added = False
        metric_added = False

        constraints = _collect_constraints_from_workflows(workflows)
        if not constraints:
            constraints = [{
                "id": f"{experiment.experiment_id}_constraint",
                "on_component": experiment.experiment_name,
                "is_hard": True,
                "how": "Auto-generated constraint"
            }]
        for constraint in constraints:
            db.session.add(ExperimentConstraint(experiment=experiment, **constraint))

        requirements = _collect_requirements_from_workflows(workflows)
        if not requirements:
            requirements = [{
                "id": f"{experiment.experiment_id}_requirement",
                "metric": experiment.intent or "quality",
                "method": "unspecified",
                "objective": "optimize"
            }]
        for requirement in requirements:
            db.session.add(ExperimentRequirement(experiment=experiment, **requirement))

        dataset_source = _pick_dataset_from_workflows(workflows)
        dataset_model = _build_dataset_from_source(experiment, dataset_source)
        if dataset_model:
            db.session.add(dataset_model)
            dataset_added = True
        elif experiment.experiment_id.startswith("exp_"):
            legacy_dataset = payload.get("variabilityPoints", {}).get("dataSet")
            dataset_model = _build_dataset_from_source(experiment, legacy_dataset)
            if dataset_model:
                db.session.add(dataset_model)
                dataset_added = True

        for model_entry in _collect_model_parameters(experiment, workflows):
            db.session.add(model_entry)
            model_added = True
        if not model_added and experiment.experiment_id.startswith("exp_"):
            legacy_model = payload.get("variabilityPoints", {}).get("model")
            if legacy_model:
                db.session.add(
                    ExperimentModel(
                        algorithm=legacy_model.get("algorithm"),
                        parameter=legacy_model.get("parameter"),
                        parameter_value=_safe_float(legacy_model.get("parameterValue")),
                        experiment=experiment,
                    )
                )
                model_added = True

        existing_metric_ids = set()
        for metric in metrics:
            db.session.add(metric)
            if metric.metric_id:
                existing_metric_ids.add(metric.metric_id)
                metric_added = True

        for metric_entry in _collect_workflow_metrics(workflows, experiment.experiment_id, existing_metric_ids):
            db.session.add(metric_entry)
            metric_added = True

        if not metric_added and experiment.experiment_id.startswith("exp_"):
            legacy_metrics = payload.get("evaluation", {}).get("runMetrics", [])
            for legacy_metric in legacy_metrics:
                metric_id = legacy_metric.get("metricId")
                if not metric_id:
                    continue
                if metric_id in existing_metric_ids:
                    continue
                existing_metric_ids.add(metric_id)
                db.session.add(
                    EvaluationMetric(
                        metric_id=metric_id,
                        experiment_id=experiment.experiment_id,
                        name=legacy_metric.get("name"),
                        value=str(legacy_metric.get("value")),
                    )
                )
                metric_added = True

        _add_placeholder_entries(experiment, dataset_added, model_added, metric_added)

    _seed_default_lessons(db)

    db.session.commit()


def load_and_insert_old(db, data_dir: str = DATA_DIR):
    """Legacy ingestion path for files/*.json experiment cards."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Legacy experiment directory not found at {data_dir}")

    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        card = data.get("experimentCard", {})
        exp_info = card.get("experimentInfo", {})
        exp_id = exp_info.get("experimentId")
        if not exp_id or Experiment.query.get(exp_id):
            continue

        start_dates = exp_info.get("experimentStartDate") or []
        end_dates = exp_info.get("experimentEndDate") or []

        experiment = Experiment(
            experiment_id=exp_id,
            experiment_name=exp_info.get("experimentName"),
            experiment_start=_parse_iso_datetime(start_dates[0]) if start_dates else None,
            experiment_end=_parse_iso_datetime(end_dates[0]) if end_dates else None,
            status=exp_info.get("status"),
            collaborators=exp_info.get("collaborators", []),
            intent=card.get("intent"),
        )
        db.session.add(experiment)

        for c in card.get("constraints", []):
            db.session.add(
                ExperimentConstraint(
                    id=c.get("id"),
                    on_component=c.get("on"),
                    is_hard=c.get("isHard"),
                    how=c.get("how"),
                    experiment=experiment,
                )
            )

        for r in card.get("requirements", []):
            db.session.add(
                ExperimentRequirement(
                    id=r.get("id"),
                    metric=r.get("metric"),
                    method=r.get("method"),
                    objective=r.get("objective"),
                    experiment=experiment,
                )
            )

        variability = card.get("variabilityPoints", {})
        if "dataSet" in variability:
            d = variability["dataSet"]
            reviewer_score = d.get("reviewer_score")
            db.session.add(
                ExperimentDataset(
                    id_dataset=d.get("id_dataset"),
                    name=d.get("name"),
                    zenoh_key_expr=d.get("zenoh_key_expr"),
                    reviewer_score=int(reviewer_score) if reviewer_score is not None else None,
                    experiment=experiment,
                )
            )

        if "model" in variability:
            m = variability["model"]
            parameter_value = m.get("parameterValue")
            db.session.add(
                ExperimentModel(
                    algorithm=m.get("algorithm"),
                    parameter=m.get("parameter"),
                    parameter_value=float(parameter_value) if parameter_value is not None else None,
                    experiment=experiment,
                )
            )

        for em in card.get("evaluation", {}).get("runMetrics", []):
            db.session.add(
                EvaluationMetric(
                    metric_id=em.get("metricId"),
                    experiment_id=experiment.experiment_id,
                    name=em.get("name"),
                    value=str(em.get("value")),
                )
            )

    _seed_default_lessons(db)

    db.session.commit()
