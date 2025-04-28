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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.