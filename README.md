# Query Form for Experiment Cards

This project is a Flask-based web application that provides a query form for collecting user feedback regarding experiment cards' evaluation. The form allows users to input relevant information, which is then processed and stored for further analysis.


## Installation

1. Clone the repository:

2. Create the docker containers:
```
docker compose up --build -d
```
3. Open your browser and navigate to `http://localhost:5002/query_example_new`.

## Usage

1. Access the query form through the web interface.
2. Fill in the fields based on which you would like to perform the filtering.
3. Click on "filter" to show only the respective fields.

## Feedback Form

1. Open your browser and navigate to `http://localhost:5002/`
2. Access the query form through the web interface.
2. Fill in the fields regarding the user feedback.
3. Click on "submit" to insert these fields as metrics linked with the respective experiment in the DAL.


## Folder Structure

```
├── static
├── files
├── templates
├── tests
├── form.py
├── requirements.txt
├── README.md
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.