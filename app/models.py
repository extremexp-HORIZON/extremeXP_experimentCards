from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Experiment(db.Model):
    __tablename__ = 'experiments'
    experiment_id = db.Column(db.String, primary_key=True)
    experiment_name = db.Column(db.String)
    experiment_start = db.Column(db.DateTime)
    experiment_end = db.Column(db.DateTime)
    status = db.Column(db.String)
    collaborators = db.Column(db.ARRAY(db.String))
    intent = db.Column(db.String)

    constraints = db.relationship("ExperimentConstraint", back_populates="experiment")
    requirements = db.relationship("ExperimentRequirement", back_populates="experiment")
    dataset = db.relationship("ExperimentDataset", back_populates="experiment", uselist=False)
    model = db.relationship("ExperimentModel", back_populates="experiment", uselist=False)
    evaluation_metrics = db.relationship("EvaluationMetric", back_populates="experiment")
    lessons = db.relationship("LessonLearnt", back_populates="experiment")


class ExperimentConstraint(db.Model):
    __tablename__ = 'experiment_constraints'
    id = db.Column(db.String)
    on_component = db.Column("on_component", db.String)
    is_hard = db.Column(db.Boolean)
    how = db.Column(db.String)
    experiment_id = db.Column(db.String, db.ForeignKey('experiments.experiment_id'))

    experiment = db.relationship("Experiment", back_populates="constraints")
    __table_args__ = (
        db.PrimaryKeyConstraint('id', 'experiment_id'),
    )

class ExperimentRequirement(db.Model):
    __tablename__ = 'experiment_requirements'
    id = db.Column(db.String)
    metric = db.Column(db.String)
    method = db.Column(db.String)
    objective = db.Column(db.String)
    experiment_id = db.Column(db.String, db.ForeignKey('experiments.experiment_id'))

    experiment = db.relationship("Experiment", back_populates="requirements")
    __table_args__ = (
        db.PrimaryKeyConstraint('id', 'experiment_id'),
    )

class ExperimentDataset(db.Model):
    __tablename__ = 'experiment_datasets'
    id_dataset = db.Column(db.String)
    name = db.Column(db.String)
    zenoh_key_expr = db.Column(db.String)
    reviewer_score = db.Column(db.Integer)
    experiment_id = db.Column(db.String, db.ForeignKey('experiments.experiment_id'))

    experiment = db.relationship("Experiment", back_populates="dataset")
    __table_args__ = (
        db.PrimaryKeyConstraint('id_dataset', 'experiment_id'),
    )



class ExperimentModel(db.Model):
    __tablename__ = 'experiment_models'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    algorithm = db.Column(db.String)
    parameter = db.Column(db.String)
    parameter_value = db.Column(db.Float)
    experiment_id = db.Column(db.String, db.ForeignKey('experiments.experiment_id'))

    experiment = db.relationship("Experiment", back_populates="model")


class EvaluationMetric(db.Model):
    __tablename__ = 'evaluation_metrics'
    metric_id = db.Column(db.String)
    name = db.Column(db.String)
    value = db.Column(db.String)
    experiment_id = db.Column(db.String, db.ForeignKey('experiments.experiment_id'))

    experiment = db.relationship("Experiment", back_populates="evaluation_metrics")
    __table_args__ = (
        db.PrimaryKeyConstraint('metric_id', 'experiment_id'),
    )

class LessonLearnt(db.Model):
    __tablename__ = 'lessons_learnt'

    lessons_learnt_id = db.Column(db.String)
    lessons_learnt = db.Column(db.String)
    experiment_rating = db.Column(db.Integer)
    run_rating = db.Column(db.ARRAY(db.Integer))
    experiment_id = db.Column(db.String, db.ForeignKey('experiments.experiment_id'))

    experiment = db.relationship("Experiment", back_populates="lessons")
    __table_args__ = (
        db.PrimaryKeyConstraint('lessons_learnt_id', 'experiment_id'),
    )
