-- Create the experiment_cards_example table
CREATE TABLE experiment_cards_example (
    experiment_id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255) NOT NULL,
    experiment_start_time TIMESTAMP NOT NULL,
    experiment_end_time TIMESTAMP NOT NULL,
    collaborators VARCHAR(255) NOT NULL,
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


-- Create the experiment_info table
CREATE TABLE experiment_info (
    experiment_info_id VARCHAR PRIMARY KEY,
    experiment_name VARCHAR,
    experiment_start_date TIMESTAMP,
    experiment_end_date TIMESTAMP,
    status VARCHAR,
    intent VARCHAR,
    collaborators VARCHAR(255)
);

-- Create the constraints table
CREATE TABLE constraints (
    constraints_id VARCHAR PRIMARY KEY,
    on_component VARCHAR,
    is_hard BOOLEAN,
    how VARCHAR
);

-- Create the requirements table
CREATE TABLE requirements (
    requirements_id VARCHAR PRIMARY KEY,
    metric VARCHAR,
    method VARCHAR,
    objective VARCHAR
);

-- Create the dataset table
CREATE TABLE dataset (
    id_dataset VARCHAR PRIMARY KEY,
    name VARCHAR,
    zenoh_key_expr VARCHAR,
    reviewer_score INTEGER
);

-- Create the model table
CREATE TABLE model (
    model_id VARCHAR PRIMARY KEY,
    algorithm VARCHAR,
    parameter VARCHAR[],
    parameter_value FLOAT[]
);

-- Create the variability_points table
CREATE TABLE variability_points (
    variability_points_id VARCHAR PRIMARY KEY,
    id_dataset VARCHAR REFERENCES dataset(id_dataset),
    model_id VARCHAR REFERENCES model(model_id)
);

-- Create the evaluation_metrics table
CREATE TABLE evaluation_metrics (
    metric_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    value VARCHAR
);

-- Create the experiment table
CREATE TABLE experiment (
    experiment_id VARCHAR PRIMARY KEY,
    experiment_info_id VARCHAR REFERENCES experiment_info(experiment_info_id),
    constraints_id VARCHAR[], 
    requirements_id VARCHAR[],
    variability_points_id VARCHAR[],
    evaluation_metrics_id VARCHAR[]
);

-- Insert mock data into the dataset table
INSERT INTO dataset (id_dataset, name, zenoh_key_expr, reviewer_score)
VALUES
('dataset_1', 'MOBI1', '1234', 80),
('dataset_2', 'MOBI2', '1235', 78),
('dataset_3', 'MOBI3', '1236', 85),
('dataset_4', 'MOBI4', '1237', 90),
('dataset_5', 'MOBI5', '1238', 88),
('dataset_6', 'MOBI6', '1239', 75),
('dataset_7', 'MOBI7', '1240', 82),
('dataset_8', 'MOBI8', '1241', 87),
('dataset_9', 'MOBI9', '1242', 89),
('dataset_10', 'MOBI10', '1243', 92);

-- Insert mock data into the model table
INSERT INTO model (model_id, algorithm, parameter, parameter_value)
VALUES
('model_1', 'RNN', ARRAY['Learning Rate'], ARRAY[0.00947]),
('model_2', 'CNN', ARRAY['Batch Size'], ARRAY[32]),
('model_3', 'SVM', ARRAY['Kernel'], ARRAY[1]),
('model_4', 'Decision Tree', ARRAY['Max Depth'], ARRAY[10]),
('model_5', 'Random Forest', ARRAY['Estimators'], ARRAY[100]),
('model_6', 'KNN', ARRAY['Neighbors'], ARRAY[5]),
('model_7', 'Logistic Regression', ARRAY['Regularization'], ARRAY[0.01]),
('model_8', 'Naive Bayes', ARRAY['Smoothing'], ARRAY[1]),
('model_9', 'Gradient Boosting', ARRAY['Learning Rate'], ARRAY[0.1]),
('model_10', 'XGBoost', ARRAY['Max Depth'], ARRAY[6]);

-- Insert mock data into the variability_points table
INSERT INTO variability_points (variability_points_id, id_dataset, model_id)
VALUES
('vp_1', 'dataset_1', 'model_1'),
('vp_2', 'dataset_2', 'model_2'),
('vp_3', 'dataset_3', 'model_3'),
('vp_4', 'dataset_4', 'model_4'),
('vp_5', 'dataset_5', 'model_5'),
('vp_6', 'dataset_6', 'model_6'),
('vp_7', 'dataset_7', 'model_7'),
('vp_8', 'dataset_8', 'model_8'),
('vp_9', 'dataset_9', 'model_9'),
('vp_10', 'dataset_10', 'model_10');

-- Insert mock data into the constraints table
INSERT INTO constraints (constraints_id, on_component, is_hard, how)
VALUES
('Constraint1', 'RNN', FALSE, 'Use'),
('Constraint2', 'CNN', TRUE, 'Optimize'),
('Constraint3', 'SVM', FALSE, 'Adjust Kernel'),
('Constraint4', 'Decision Tree', TRUE, 'Limit Depth'),
('Constraint5', 'Random Forest', TRUE, 'Increase Estimators'),
('Constraint6', 'KNN', FALSE, 'Adjust Neighbors'),
('Constraint7', 'Logistic Regression', TRUE, 'Regularize'),
('Constraint8', 'Naive Bayes', FALSE, 'Adjust Smoothing'),
('Constraint9', 'Gradient Boosting', TRUE, 'Tune Learning Rate'),
('Constraint10', 'XGBoost', TRUE, 'Optimize Depth');

-- Insert mock data into the requirements table
INSERT INTO requirements (requirements_id, metric, method, objective)
VALUES
('Requirement1', 'MSE', 'TrainTestSplit', 'Minimize'),
('Requirement2', 'Accuracy', 'CrossValidation', 'Maximize'),
('Requirement3', 'Precision', 'CrossValidation', 'Maximize'),
('Requirement4', 'Recall', 'TrainTestSplit', 'Maximize'),
('Requirement5', 'F1-Score', 'CrossValidation', 'Maximize'),
('Requirement6', 'ROC-AUC', 'TrainTestSplit', 'Maximize'),
('Requirement7', 'Log-Loss', 'CrossValidation', 'Minimize'),
('Requirement8', 'MAE', 'TrainTestSplit', 'Minimize'),
('Requirement9', 'RMSE', 'CrossValidation', 'Minimize'),
('Requirement10', 'R-Squared', 'TrainTestSplit', 'Maximize');

-- Insert mock data into the evaluation_metrics table
INSERT INTO evaluation_metrics (metric_id, name, value)
VALUES
('metric_1', 'MSE', '0.02502'),
('metric_2', 'RMSE', '0.15818'),
('metric_3', 'R-Squared', '0.9'),
('metric_4', 'MAE', '0.02203'),
('metric_5', 'Accuracy', '0.95'),
('metric_6', 'Precision', '0.92'),
('metric_7', 'Recall', '0.89'),
('metric_8', 'F1-Score', '0.91'),
('metric_9', 'ROC-AUC', '0.94'),
('metric_10', 'Log-Loss', '0.05');

-- Insert mock data into the experiment_info table
INSERT INTO experiment_info (experiment_info_id, experiment_name, experiment_start_date, experiment_end_date, status, intent, collaborators)
VALUES
('exp_info_1', 'Experiment_1', '2025-01-02 10:00:00', '2025-01-02 11:15:00', 'completed', 'classification', 'User_B'),
('exp_info_2', 'Experiment_2', '2025-02-01 09:00:00', '2025-02-15 09:01:00', 'ongoing', 'classification', 'User_C'),
('exp_info_3', 'Experiment_3', '2025-03-01 08:30:00', '2025-03-20 16:30:00', 'completed', 'regression', 'User_D'),
('exp_info_4', 'Experiment_4', '2025-04-01 11:00:00', '2025-04-10 15:00:00', 'failed', 'clustering', 'User_E'),
('exp_info_5', 'Experiment_5', '2025-05-01 14:00:00', '2025-05-20 19:00:00', 'completed', 'classification', 'User_F'),
('exp_info_6', 'Experiment_6', '2025-06-01 10:00:00', '2025-06-15 12:00:00', 'ongoing', 'regression','User_G'),
('exp_info_7', 'Experiment_7', '2025-07-01 09:00:00', '2025-07-10 11:00:00', 'completed', 'clustering', 'User_H'),
('exp_info_8', 'Experiment_8', '2025-08-01 08:00:00', '2025-08-20 10:00:00', 'failed', 'clustering', 'User_I'),
('exp_info_9', 'Experiment_9', '2025-09-01 07:00:00', '2025-09-15 09:00:00', 'completed', 'regression','User_J'),
('exp_info_10', 'Experiment_10', '2025-10-01 06:00:00', '2025-10-20 08:00:00', 'ongoing', 'regression','User_K');

-- Insert mock data into the experiment table
INSERT INTO experiment (experiment_id, experiment_info_id, constraints_id, requirements_id, variability_points_id, evaluation_metrics_id)
VALUES
('exp_1', 'exp_info_1', ARRAY['Constraint1'], ARRAY['Requirement1'], ARRAY['vp_1'], ARRAY['metric_1', 'metric_2', 'metric_3', 'metric_4']),
('exp_2', 'exp_info_2', ARRAY['Constraint2'], ARRAY['Requirement2'], ARRAY['vp_2'], ARRAY['metric_5']),
('exp_3', 'exp_info_3', ARRAY['Constraint3'], ARRAY['Requirement3'], ARRAY['vp_3'], ARRAY['metric_6']),
('exp_4', 'exp_info_4', ARRAY['Constraint4'], ARRAY['Requirement4'], ARRAY['vp_4'], ARRAY['metric_7']),
('exp_5', 'exp_info_5', ARRAY['Constraint5'], ARRAY['Requirement5'], ARRAY['vp_5'], ARRAY['metric_8']),
('exp_6', 'exp_info_6', ARRAY['Constraint6'], ARRAY['Requirement6'], ARRAY['vp_6'], ARRAY['metric_9']),
('exp_7', 'exp_info_7', ARRAY['Constraint7'], ARRAY['Requirement7'], ARRAY['vp_7'], ARRAY['metric_10']),
('exp_8', 'exp_info_8', ARRAY['Constraint8'], ARRAY['Requirement8'], ARRAY['vp_8'], ARRAY['metric_1']),
('exp_9', 'exp_info_9', ARRAY['Constraint9'], ARRAY['Requirement9'], ARRAY['vp_9'], ARRAY['metric_2']),
('exp_10', 'exp_info_10', ARRAY['Constraint10'], ARRAY['Requirement10'], ARRAY['vp_10'], ARRAY['metric_3']);