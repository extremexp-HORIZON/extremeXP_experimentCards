import psycopg2
import random
from datetime import datetime, timedelta

# Database connection
conn = psycopg2.connect(
    dbname="experiment_db",
    user="user",
    password="password",
    host="localhost",
    port="5433"
)
cursor = conn.cursor()

# Helper function to generate random dates
def random_date(start, end):
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)

# Insert mock data into experiment_info
def insert_experiment_info():
    for i in range(1, 6):  # Insert 5 mock records
        experiment_id = f"exp_{i}"
        experiment_name = f"Experiment {i}"
        start_date = random_date(datetime(2023, 1, 1), datetime(2023, 12, 31))
        end_date = start_date + timedelta(days=random.randint(1, 30))
        status = random.choice(["Completed", "Ongoing", "Failed"])
        collaborators = ["Alice", "Bob", "Charlie"]
        
        cursor.execute("""
            INSERT INTO experiment_info (experiment_id, experiment_name, experiment_start_date, experiment_end_date, status, collaborators)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            experiment_id,
            experiment_name,
            start_date,
            end_date,
            status,
            collaborators
        ))
        print(f"Inserted into experiment_info: {experiment_id}")

# Insert mock data into constraints
def insert_constraints():
    for i in range(1, 6):  # Insert 5 mock records
        constraints_id = f"constraint_{i}"
        on_component = f"Component {i}"
        is_hard = random.choice([True, False])
        how = f"Constraint description {i}"
        
        cursor.execute("""
            INSERT INTO constraints (constraints_id, on_component, is_hard, how)
            VALUES (%s, %s, %s, %s)
        """, (
            constraints_id,
            on_component,
            is_hard,
            how
        ))
        print(f"Inserted into constraints: {constraints_id}")

# Insert mock data into requirements
def insert_requirements():
    for i in range(1, 6):  # Insert 5 mock records
        requirements_id = f"requirement_{i}"
        metric = f"Metric {i}"
        method = f"Method {i}"
        objective = f"Objective {i}"
        
        cursor.execute("""
            INSERT INTO requirements (requirements_id, metric, method, objective)
            VALUES (%s, %s, %s, %s)
        """, (
            requirements_id,
            metric,
            method,
            objective
        ))
        print(f"Inserted into requirements: {requirements_id}")

# Insert mock data into variability_points
def insert_variability_points():
    for i in range(1, 6):  # Insert 5 mock records
        variability_points_id = f"vp_{i}"
        id_dataset = f"dataset_{i}"
        model_id = f"model_{i}"
        
        cursor.execute("""
            INSERT INTO variability_points (variability_points_id, id_dataset, model_id)
            VALUES (%s, %s, %s)
        """, (
            variability_points_id,
            id_dataset,
            model_id
        ))
        print(f"Inserted into variability_points: {variability_points_id}")

# Insert mock data into evaluation_metrics
def insert_evaluation_metrics():
    for i in range(1, 6):  # Insert 5 mock records
        metric_id = f"metric_{i}"
        name = f"Metric Name {i}"
        value = f"Value {i}"
        
        cursor.execute("""
            INSERT INTO evaluation_metrics (metric_id, name, value)
            VALUES (%s, %s, %s)
        """, (
            metric_id,
            name,
            value
        ))
        print(f"Inserted into evaluation_metrics: {metric_id}")

# Insert mock data into experiment
def insert_experiment():
    for i in range(1, 6):  # Insert 5 mock records
        experiment_id = f"exp_{i}"
        constraints_id = f"constraint_{i}"
        requirements_id = f"requirement_{i}"
        variability_points_id = f"vp_{i}"
        evaluation_metrics_id = f"metric_{i}"
        
        cursor.execute("""
            INSERT INTO experiment (experiment_id, constraints_id, requirements_id, variability_points_id, evaluation_metrics_id)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            experiment_id,
            constraints_id,
            requirements_id,
            variability_points_id,
            evaluation_metrics_id
        ))
        print(f"Inserted into experiment: {experiment_id}")

# Insert mock data into all tables
try:
    insert_experiment_info()
    insert_constraints()
    insert_requirements()
    insert_variability_points()
    insert_evaluation_metrics()
    insert_experiment()
    conn.commit()
    print("Mock data inserted successfully!")
except Exception as e:
    conn.rollback()
    print(f"An error occurred: {e}")
finally:
    cursor.close()
    conn.close()