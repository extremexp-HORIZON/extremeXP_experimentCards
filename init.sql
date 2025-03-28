-- Create the experiment_cards_example table
CREATE TABLE experiment_cards_example (
    experiment_id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255) NOT NULL,
    experiment_start_time TIMESTAMP NOT NULL,
    experiment_end_time TIMESTAMP NOT NULL,
    collaborators TEXT NOT NULL,
    intent VARCHAR(50) NOT NULL CHECK (intent IN ('classification', 'regression', 'clustering'))
);

-- Insert mock data into experiment_cards_example
INSERT INTO experiment_cards_example (experiment_name, experiment_start_time, experiment_end_time, collaborators, intent)
VALUES
('Experiment 1', '2025-01-01 10:00:00', '2025-01-10 11:00:00', 'Alice Johnson', 'classification'),
('Experiment 2', '2025-02-01 09:00:00', '2025-02-15 09:01:00', 'Diana Prince', 'regression'),
('Experiment 3', '2025-03-01 08:30:00', '2025-03-20 16:30:00',  'Frank White', 'clustering'),
('Experiment 4', '2025-04-01 11:00:00', '2025-04-10 15:00:00', 'Hank Green', 'classification'),
('Experiment 5', '2025-05-01 14:00:00', '2025-05-20 19:00:00', 'Jack Black', 'regression');

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
