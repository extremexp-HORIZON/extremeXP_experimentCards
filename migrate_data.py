import os
import json
import psycopg2
from psycopg2 import errors

# Database connection
conn = psycopg2.connect(
    dbname="experiment_db",
    user="user",
    password="password",
    host="localhost",  
    port="5433"
)
cursor = conn.cursor()

# Directory containing experiment JSON files
directory = './files/'

# Loop through all JSON files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.json'):  # Process only JSON files
        filepath = os.path.join(directory, filename)
        print(f"Processing file: {filename}")
        try:
            with open(filepath) as f:
                data = json.load(f)
                print(data)

            # Insert experiment_info
            experiment_info = data['experimentCard']['experimentInfo']
            print("hello1")
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
            print("hello2")
            # Insert constraints
            for constraint in data['experimentCard']['constraints']:
                cursor.execute("""
                    INSERT INTO constraints (constraints_id, on_component, is_hard, how)
                    VALUES (%s, %s, %s, %s)
                """, (
                    constraint['id'],
                    constraint['on'],
                    constraint['isHard'],
                    constraint['how']
                ))

            # Insert requirements
            for requirement in data['experimentCard']['requirements']:
                cursor.execute("""
                    INSERT INTO requirements (requirements_id, metric, method, objective)
                    VALUES (%s, %s, %s, %s)
                """, (
                    requirement['id'],
                    requirement['metric'],
                    requirement['method'],
                    requirement['objective']
                ))

            # Insert dataset
            dataset = data['experimentCard']['variabilityPoints']['dataSet']
            cursor.execute("""
                INSERT INTO dataset (id_dataset, name, zenoh_key_expr, reviewer_score)
                VALUES (%s, %s, %s, %s)
            """, (
                dataset['id_dataset'],
                dataset['name'],
                dataset['zenoh_key_expr'],
                dataset['reviewer_score']
            ))

            # Insert variability_points
            variability_points = data['experimentCard']['variabilityPoints']
            cursor.execute("""
                INSERT INTO variability_points (variability_points_id, id_dataset, model_id)
                VALUES (%s, %s, %s)
            """, (
                variability_points['id'],
                variability_points['dataSet']['id_dataset'],
                variability_points['model']['id']
            ))

            # Insert model
            model = variability_points['model']
            cursor.execute("""
                INSERT INTO model (model_id, algorithm, parameter, parameter_value)
                VALUES (%s, %s, %s, %s)
            """, (
                model['id'],
                model['algorithm'],
                model['parameter'],
                model['parameterValue']
            ))

            # Insert evaluation_metrics
            for metric in data['experimentCard']['evaluation']['runMetrics']:
                cursor.execute("""
                    INSERT INTO evaluation_metrics (metric_id, name, value)
                    VALUES (%s, %s, %s)
                """, (
                    metric['metricId'],
                    metric['name'],
                    metric['value']
                ))

            # Insert experiment
            cursor.execute("""
                INSERT INTO experiment (experiment_id, constraints_id, requirements_id, variability_points_id, evaluation_metrics_id)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                experiment_info['experimentId'],
                constraint['id'] if data['experimentCard']['constraints'] else None,
                requirement['id'] if data['experimentCard']['requirements'] else None,
                variability_points['id'],
                metric['metricId'] if data['experimentCard']['evaluation']['runMetrics'] else None
            ))

        except errors.UniqueViolation as e:
            print(f"Duplicate entry found in file {filename}: {e}")
            conn.rollback()  # Roll back the current transaction to avoid locking issues
        except Exception as e:
            print(f"An error occurred while processing file {filename}: {e}")
            conn.rollback()
        else:
            conn.commit()  # Commit the transaction if no errors occurred

# Close the connection
cursor.close()
conn.close()