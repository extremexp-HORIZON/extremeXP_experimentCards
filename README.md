#  ExtremeXP Experiment Cards Component

This is a Flask-based web application designed for Experiment Card management, query and user feedback insertion. The project includes Swagger integration for API documentation and front-end interfaces for user interaction (a form for lessons learnt user feedback for expeiments and a page for viewing, managing and querying the experiment cards).

## Features

- **Flask Framework**: A lightweight and flexible Python web framework.
- **Swagger Integration**: API documentation using Swagger.
- **Docker Support**: Easily containerize and deploy the application.

---

## Prerequisites

Before running the application, ensure you have the Docker installed.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/XXXX/flask-extremeXP.git
   cd flask-extremeXP

2. **Set Up Environment Variables**:
   Create a `.env` file in the project root and add the following:
   ```properties
   ACCESS_TOKEN=your_access_token_here
   ```

3. **Build the Docker Image**:
   Use Docker Compose to build the application image:
   ```bash
   docker-compose build
   ```

4. **Run the Application**:
   Start the application using Docker Compose:
   ```bash
   docker-compose up
   ```

5. **Access the Application**:
   Once the application is running, you can view and query experiment cards in your browser at:
   ```
   http://127.0.0.1:5002/query_experiments_page 
   ```
    Moreover, to insert user feedback and lessons learnt to an existing experiment with id {experiment_id}, type in your browser and fill in the form:
    ```
   http://127.0.0.1:5002/form_lessons_learnt/{experiment_id}
   ```
6. **View API Documentation**:
   The Swagger UI for API documentation is available at:
   ```
   http://127.0.0.1:5002/swagger
   ```

---

## Front-End Screenshots
One core interface is a form-based feedback submission page, designed to collect user feedback immediately after the execution of an experiment
<img width="220" height="283" alt="image" src="https://github.com/user-attachments/assets/a7f8185d-2f32-4ceb-9fa9-39242f7483d3" />


Complementing this form, we have developed a centralized page for Experiment Cards, where users can browse, search, and filter documented experiments 
<img width="280" height="198" alt="image" src="https://github.com/user-attachments/assets/ea7e89f0-6919-4292-a093-c6a2b6a01611" />



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
