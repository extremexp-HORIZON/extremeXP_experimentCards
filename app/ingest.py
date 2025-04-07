import os
import json
from .models import Experiment, ExperimentConstraint, ExperimentRequirement, ExperimentDataset, ExperimentModel, EvaluationMetric

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'files')

def load_and_insert(db):
        for file in os.listdir(DATA_DIR):
            if file.endswith(".json"):
                path = os.path.join(DATA_DIR, file)
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    card = data["experimentCard"]
                    print(data)

                    exp_info = card["experimentInfo"]
                    experiment = Experiment(
                        experiment_id=exp_info["experimentId"],
                        experiment_name=exp_info["experimentName"],
                        experiment_start=exp_info["experimentStartDate"][0],
                        experiment_end=exp_info["experimentEndDate"][0],
                        status=exp_info["status"],
                        collaborators=exp_info.get("collaborators", [])
                    )
                    db.session.add(experiment)
                    print(experiment)

                    # # Constraints
                    # for c in card.get("constraints", []):
                    #     db.session.add(ExperimentConstraint(
                    #         id=c["id"],
                    #         on_component=c["on"],
                    #         is_hard=c["isHard"],
                    #         how=c["how"],
                    #         experiment=experiment
                    #     ))
                    from sqlalchemy.dialects.postgresql import insert

                    for c in card.get("constraints", []):
                        stmt = insert(ExperimentConstraint).values(id=c["id"], on_component=c["on"], is_hard=c["isHard"], how=c["how"], experiment_id=experiment.experiment_id
                                                                   ).on_conflict_do_update(
                                                                         index_elements=['id'],  # Primary key column
                                                                         set_={"on_component": c["on"], "is_hard": c["isHard"], "how": c["how"] })
                        db.session.execute(stmt)

                    # # Requirements
                    # for r in card.get("requirements", []):
                    #     db.session.add(ExperimentRequirement(
                    #         id=r["id"],
                    #         metric=r["metric"],
                    #         method=r["method"],
                    #         objective=r["objective"],
                    #         experiment=experiment
                    #     ))
                    #     db.session.execute(stmt)
                   
                    # # Variability - Dataset
                    # if "dataSet" in card.get("variabilityPoints", {}):
                    #     d = card["variabilityPoints"]["dataSet"]
                    #     db.session.add(ExperimentDataset(
                    #         id_dataset=d["id_dataset"],
                    #         name=d["name"],
                    #         zenoh_key_expr=d["zenoh_key_expr"],
                    #         reviewer_score=int(d["reviewer_score"]),
                    #         experiment=experiment
                    #     ))

                    # # Variability - Model
                    # if "model" in card.get("variabilityPoints", {}):
                    #     m = card["variabilityPoints"]["model"]
                    #     db.session.add(ExperimentModel(
                    #         algorithm=m["algorithm"],
                    #         parameter=m["parameter"],
                    #         parameter_value=float(m["parameterValue"]),
                    #         experiment=experiment
                    #     ))

    #                 for em in card["evaluation"].get("runMetrics", []):
    # existing_metric = EvaluationMetric.query.filter_by(metric_id=em["metricId"]).first()
    # if existing_metric:
    #     existing_metric.name = em["name"]
    #     existing_metric.value = float(em["value"])
    # else:
    #     db.session.add(EvaluationMetric(
    #         metric_id=em["metricId"],
    #         name=em["name"],
    #         value=float(em["value"]),
    #         experiment=experiment
    #     ))

        db.session.commit()
