import os
import json
from .models import Experiment, ExperimentConstraint, ExperimentRequirement, ExperimentDataset, ExperimentModel, EvaluationMetric, LessonLearnt
import random

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
                        collaborators=exp_info.get("collaborators", []),
                        intent = card["intent"]
                    )
                    db.session.add(experiment)
                    print(experiment)

                    for c in card.get("constraints", []):
                        db.session.add(ExperimentConstraint(
                            id=c["id"],
                            on_component=c["on"],
                            is_hard=c["isHard"],
                            how=c["how"],
                            experiment=experiment
                        ))

                    for r in card.get("requirements", []):
                        db.session.add(ExperimentRequirement(
                            id=r["id"],
                            metric=r["metric"],
                            method=r["method"],
                            objective=r["objective"],
                            experiment=experiment
                        ))
                   
                    if "dataSet" in card.get("variabilityPoints", {}):
                        d = card["variabilityPoints"]["dataSet"]
                        db.session.add(ExperimentDataset(
                            id_dataset=d["id_dataset"],
                            name=d["name"],
                            zenoh_key_expr=d["zenoh_key_expr"],
                            reviewer_score=int(d["reviewer_score"]),
                            experiment=experiment
                        ))

                    if "model" in card.get("variabilityPoints", {}):
                        m = card["variabilityPoints"]["model"]
                        db.session.add(ExperimentModel(
                            algorithm=m["algorithm"],
                            parameter=m["parameter"],
                            parameter_value=float(m["parameterValue"]),
                            experiment=experiment
                        ))

                    for em in card["evaluation"].get("runMetrics", []):
                        db.session.add(EvaluationMetric(
                            metric_id=em["metricId"],
                            experiment_id=experiment.experiment_id,
                            name=em["name"],
                            value=em["value"],
                        ))

        for i in range(1, 21):
            exp_id = f"exp_{i}"
            lesson_id = f"lessons_learnt_{i}"
            lesson_text = f"Lessons Learnt {i} text"
            rating = random.randint(1, 7)
            run_ratings = [random.randint(1, 10) for _ in range(random.randint(1, 4))]

            db.session.add(LessonLearnt(
                lessons_learnt_id=lesson_id,
                lessons_learnt=lesson_text,
                experiment_rating=rating,
                run_rating=run_ratings,
                experiment_id=exp_id
            ))      

        db.session.commit()
