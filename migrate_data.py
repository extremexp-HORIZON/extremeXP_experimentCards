import json
import psycopg2

with open('./files/experiment_1.json') as f:
    data = json.load(f)

conn = psycopg2.connect(
    dbname="experiment_db",
    user="user",
    password="password",
    host="localhost",  
    port="5432"
)
cursor = conn.cursor()

experiment_info = data['experimentCard']['experimentInfo']
cursor.execute("""
    INSERT INTO experiment_info (experiment_id, experiment_name, experiment_start_date, experiment_end_date, status, collaborators)
    VALUES (%s, %s, %s, %s, %s, %s)
""", (
    experiment_info['experimentId'],
    experiment_info['experimentName'],
    experiment_info['experimentStartDate'][0],
    experiment_info['experimentEndDate'][0],
    experiment_info['status'],
    experiment_info['collaborators']
))

for constraint in data['experimentCard']['constraints']:
    cursor.execute("""
        INSERT INTO constraints (id, on_component, is_hard, how)
        VALUES (%s, %s, %s, %s)
    """, (
        constraint['id'],
        constraint['on'],
        constraint['isHard'],
        constraint['how']
    ))

for requirement in data['experimentCard']['requirements']:
    cursor.execute("""
        INSERT INTO requirements (id, metric, method, objective)
        VALUES (%s, %s, %s, %s)
    """, (
        requirement['id'],
        requirement['metric'],
        requirement['method'],
        requirement['objective']
    ))

variability_points = data['experimentCard']['variabilityPoints']['dataSet']
cursor.execute("""
    INSERT INTO variability_points (id_dataset, name, zenoh_key_expr, reviewer_score)
    VALUES (%s, %s, %s, %s)
""", (
    variability_points['id_dataset'],
    variability_points['name'],
    variability_points['zenoh_key_expr'],
    variability_points['reviewer_score']
))

model = data['experimentCard']['variabilityPoints']['model']
cursor.execute("""
    INSERT INTO model (algorithm, parameter, parameter_value)
    VALUES (%s, %s, %s)
""", (
    model['algorithm'],
    model['parameter'],
    model['parameterValue']
))

for metric in data['experimentCard']['evaluation']['runMetrics']:
    cursor.execute("""
        INSERT INTO evaluation_metrics (metric_id, name, value)
        VALUES (%s, %s, %s)
    """, (
        metric['metricId'],
        metric['name'],
        metric['value']
    ))

conn.commit()
cursor.close()
conn.close()
