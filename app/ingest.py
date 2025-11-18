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

import re
from itertools import count

_CONSTRAINT_RX = re.compile(r'//\s*constraint\s*:\s*(?P<on>\w+)\s*-\s*(?P<how>.+)', re.IGNORECASE)
_REQUIREMENT_RX = re.compile(
    r'//\s*requirement\s*:\s*(?P<metric>\w+)\s*\((?P<objective>.+?)\)\s*via\s*(?P<method>.+)',
    re.IGNORECASE,
)
_DATASET_RX = re.compile(r'//\s*dataset\s*:\s*(?P<name>[^|]+)\|\s*key=(?P<key>[^\s|]+)\s*\|\s*score=(?P<score>.+)',
                         re.IGNORECASE)
_MODEL_RX = re.compile(r'//\s*model\s*:\s*(?P<algo>\w+)\s*\((?P<param>[^=]+)=(?P<value>.+)\)', re.IGNORECASE)

def _extract_constraints(model_text: str):
    for idx, match in zip(count(1), _CONSTRAINT_RX.finditer(model_text or "")):
        yield {
            "id": f"auto_constraint_{idx}",
            "on_component": match.group("on"),
            "is_hard": True,
            "how": match.group("how").strip(),
        }

def _extract_requirements(model_text: str):
    for idx, match in zip(count(1), _REQUIREMENT_RX.finditer(model_text or "")):
        yield {
            "id": f"auto_requirement_{idx}",
            "metric": match.group("metric"),
            "method": match.group("method").strip(),
            "objective": match.group("objective").strip(),
        }

def _extract_dataset(model_text: str):
    match = _DATASET_RX.search(model_text or "")
    if not match:
        return None
    score = match.group("score").strip()
    return {
        "id_dataset": f"auto_dataset_{match.group('name').strip().lower().replace(' ', '_')}",
        "name": match.group("name").strip(),
        "zenoh_key_expr": match.group("key").strip(),
        "reviewer_score": int(score) if score.isdigit() else None,
    }

def _extract_model_variability(model_text: str):
    match = _MODEL_RX.search(model_text or "")
    if not match:
        return None
    try:
        param_value = float(match.group("value"))
    except ValueError:
        param_value = None
    return {
        "algorithm": match.group("algo"),
        "parameter": match.group("param").strip(),
        "parameter_value": param_value,
    }

def _extract_metric_value(metric_payload: Dict[str, Any]):
    if not metric_payload:
        return None
    if "value" in metric_payload and metric_payload["value"] is not None:
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

def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
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

def _pick_dataset_from_workflows(workflows: List[Dict[str, Any]]):
    for workflow in workflows:
        inputs = workflow.get("input_datasets") or []
        if inputs:
            return inputs[0]
        outputs = workflow.get("output_datasets") or []
        if outputs:
            return outputs[0]
    return None

def _build_dataset_from_source(experiment: Experiment, dataset_source: Dict[str, Any]):
    if not dataset_source:
        return None
    dataset_id = dataset_source.get("id") or dataset_source.get("id_dataset") or dataset_source.get("name") or dataset_source.get("uri")
    if not dataset_id:
        return None
    reviewer_score = dataset_source.get("reviewer_score")
    if reviewer_score is not None:
        try:
            reviewer_score = int(reviewer_score)
        except (TypeError, ValueError):
            reviewer_score = None
    return ExperimentDataset(
        id_dataset=dataset_id,
        name=dataset_source.get("name"),
        zenoh_key_expr=dataset_source.get("uri") or dataset_source.get("zenoh_key_expr"),
        reviewer_score=reviewer_score,
        experiment=experiment,
    )

def _collect_model_parameters(experiment: Experiment, workflows: List[Dict[str, Any]]) -> List[ExperimentModel]:
    entries = []
    for workflow in workflows:
        algorithm = workflow.get("name")
        for param in workflow.get("parameters", []) or []:
            entries.append(
                ExperimentModel(
                    algorithm=algorithm,
                    parameter=param.get("name"),
                    parameter_value=_safe_float(param.get("value")),
                    experiment=experiment,
                )
            )
        for task in workflow.get("tasks", []) or []:
            task_algorithm = task.get("name") or algorithm
            for param in task.get("parameters", []) or []:
                entries.append(
                    ExperimentModel(
                        algorithm=task_algorithm,
                        parameter=param.get("name"),
                        parameter_value=_safe_float(param.get("value")),
                        experiment=experiment,
                    )
                )
    return entries

def _collect_workflow_metrics(workflows: List[Dict[str, Any]], experiment_id: str, existing_ids: set) -> List[EvaluationMetric]:
    collected = []
    for workflow in workflows:
        workflow_id = workflow.get("id")
        collected.extend(_build_metric_rows(workflow_id, workflow.get("metrics"), experiment_id, existing_ids))
        for task in workflow.get("tasks", []) or []:
            prefix = f"{workflow_id}_{task.get('id')}" if workflow_id else task.get("id")
            collected.extend(_build_metric_rows(prefix, task.get("metrics"), experiment_id, existing_ids))
    return collected

def _build_metric_rows(prefix: str, metrics_source: Any, experiment_id: str, existing_ids: set) -> List[EvaluationMetric]:
    rows = []
    if not metrics_source:
        return rows
    for idx, metric in enumerate(metrics_source, start=1):
        if not isinstance(metric, dict):
            continue
        metric_id = metric.get("id") or f"{prefix}_{idx}" if prefix else f"metric_{idx}"
        original_id = metric_id
        increment = idx
        while metric_id in existing_ids:
            increment += 1
            metric_id = f"{original_id}_{increment}"
        existing_ids.add(metric_id)
        value = metric.get("value")
        if value is None:
            records = metric.get("records")
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
                name=metric.get("name"),
                value=str(value) if value is not None else None,
            )
        )
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

def _parse_iso_datetime(value: str):
    """Convert ISO8601 strings (with optional trailing Z) to datetime objects."""
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None

def _map_experiment_payload(payload: Dict[str, Any], metrics_index: Dict[str, Any]) -> Tuple[Experiment, List[EvaluationMetric], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
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

    metrics = [
        EvaluationMetric(
            metric_id=metric_id,
            experiment_id=experiment.experiment_id,
            name=(metrics_index.get(metric_id, {}) or {}).get("name") or f"metric_{index}",
            value=_extract_metric_value(metrics_index.get(metric_id, {})),
        )
        for index, metric_id in enumerate(payload.get("metric_ids", []), start=1)
    ]

    model_text = payload.get("model", "") or ""

    constraints = list(_extract_constraints(model_text))
    requirements = list(_extract_requirements(model_text))
    dataset = _extract_dataset(model_text)
    model_cfg = _extract_model_variability(model_text)

    return experiment, metrics, constraints, requirements, dataset, model_cfg

def _seed_default_lessons(db):
    """Populate demo lessons for quick testing."""
    for i in range(1, 21):
        exp_id = f"exp_{i}"
        lesson_id = f"lessons_learnt_{i}"
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

    for entry in experiments_data:
        if not isinstance(entry, dict):
            continue
        exp_id, payload = next(iter(entry.items()))
        if Experiment.query.get(exp_id):
            continue

        experiment, metrics, constraints, requirements, dataset_cfg, model_cfg = _map_experiment_payload(payload, metrics_index)
        db.session.add(experiment)

        for constraint in constraints:
            db.session.add(ExperimentConstraint(experiment=experiment, **constraint))

        for requirement in requirements:
            db.session.add(ExperimentRequirement(experiment=experiment, **requirement))

        if dataset_cfg and not experiment.dataset:
            db.session.add(ExperimentDataset(experiment=experiment, **dataset_cfg))

        if model_cfg:
            db.session.add(ExperimentModel(experiment=experiment, **model_cfg))

        workflows = _get_workflows_for_experiment(payload, workflow_index, workflow_groups)

        inferred_start, inferred_end = _infer_timings_from_workflows(workflows)
        if experiment.experiment_start is None and inferred_start:
            experiment.experiment_start = inferred_start
        if experiment.experiment_end is None and inferred_end:
            experiment.experiment_end = inferred_end

        existing_metric_ids = set()
        for metric in metrics:
            db.session.add(metric)
            if metric.metric_id:
                existing_metric_ids.add(metric.metric_id)

        dataset_source = _pick_dataset_from_workflows(workflows)
        if dataset_source and not experiment.dataset:
            dataset_model = _build_dataset_from_source(experiment, dataset_source)
            if dataset_model:
                db.session.add(dataset_model)

        for model_entry in _collect_model_parameters(experiment, workflows):
            db.session.add(model_entry)

        for metric_entry in _collect_workflow_metrics(workflows, experiment.experiment_id, existing_metric_ids):
            db.session.add(metric_entry)

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
