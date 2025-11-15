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

EXPERIMENTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "dal_files",
    "experiments.json",
)
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

def _parse_iso_datetime(value: str):
    """Convert ISO8601 strings (with optional trailing Z) to datetime objects."""
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None

def _map_experiment_payload(payload: Dict[str, Any]) -> Tuple[Experiment, List[EvaluationMetric]]:
    """Build ORM objects (experiment + metrics) from a raw experiments.json payload."""
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
            name=f"metric_{index}",
            value=str(metric_id),
        )
        for index, metric_id in enumerate(payload.get("metric_ids", []), start=1)
    ]

    model_text = payload.get("model", "") or ""

    # Heuristic parsing from the model DSL
    for constraint in _extract_constraints(model_text):
        db.session.add(ExperimentConstraint(experiment=experiment, **constraint))

    for requirement in _extract_requirements(model_text):
        db.session.add(ExperimentRequirement(experiment=experiment, **requirement))

    dataset = _extract_dataset(model_text)
    if dataset:
        db.session.add(ExperimentDataset(experiment=experiment, **dataset))

    model_cfg = _extract_model_variability(model_text)
    if model_cfg:
        db.session.add(ExperimentModel(experiment=experiment, **model_cfg))

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

    with open(EXPERIMENTS_PATH, "r", encoding="utf-8") as f:
        experiments_data = json.load(f).get("experiments", [])

    for entry in experiments_data:
        if not isinstance(entry, dict):
            continue
        exp_id, payload = next(iter(entry.items()))
        if Experiment.query.get(exp_id):
            continue

        experiment, metrics = _map_experiment_payload(payload)
        db.session.add(experiment)

        for metric in metrics:
            db.session.add(metric)

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
