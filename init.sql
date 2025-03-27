CREATE TABLE experiment (
    experiment_id VARCHAR PRIMARY KEY,
    constraints_id VARCHAR,
    requirements_id VARCHAR,
    variability_points_id VARCHAR,
    evaluation_metrics_id VARCHAR
);

CREATE TABLE experiment_info (
    experiment_id VARCHAR PRIMARY KEY,
    experiment_name VARCHAR,
    experiment_start_date TIMESTAMP,
    experiment_end_date TIMESTAMP,
    status VARCHAR,
    collaborators TEXT[]
);

CREATE TABLE constraints (
    constraints_id VARCHAR PRIMARY KEY,
    on_component VARCHAR,
    is_hard BOOLEAN,
    how VARCHAR
);

CREATE TABLE requirements (
    requirements_id VARCHAR PRIMARY KEY,
    metric VARCHAR,
    method VARCHAR,
    objective VARCHAR
);

CREATE TABLE dataset (
    id_dataset VARCHAR ,
    name VARCHAR,
    zenoh_key_expr VARCHAR,
    reviewer_score INTEGER
);

CREATE TABLE variability_points (
    variability_points_id VARCHAR PRIMARY KEY,
    id_dataset VARCHAR,
    model_id VARCHAR
);

CREATE TABLE model (
    model_id VARCHAR PRIMARY KEY,
    algorithm VARCHAR,
    parameter VARCHAR,
    parameter_value FLOAT
);

CREATE TABLE evaluation_metrics (
    metric_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    value VARCHAR
);
