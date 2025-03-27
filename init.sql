CREATE TABLE experiment_info (
    experiment_id VARCHAR PRIMARY KEY,
    experiment_name VARCHAR,
    experiment_start_date TIMESTAMP,
    experiment_end_date TIMESTAMP,
    status VARCHAR,
    collaborators TEXT[]
);

CREATE TABLE constraints (
    id VARCHAR PRIMARY KEY,
    on_component VARCHAR,
    is_hard BOOLEAN,
    how VARCHAR
);

CREATE TABLE requirements (
    id VARCHAR PRIMARY KEY,
    metric VARCHAR,
    method VARCHAR,
    objective VARCHAR
);

CREATE TABLE variability_points (
    id_dataset VARCHAR PRIMARY KEY,
    name VARCHAR,
    zenoh_key_expr VARCHAR,
    reviewer_score INTEGER
);

CREATE TABLE model (
    algorithm VARCHAR,
    parameter VARCHAR,
    parameter_value FLOAT
);

CREATE TABLE evaluation_metrics (
    metric_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    value FLOAT
);
