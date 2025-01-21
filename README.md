# Predictive Model Elicitation

This project is designed to elicit information from users and update a predictive model based on their responses. It uses OpenAI's GPT-4 model to generate questions, predict answers, and update the model.

## Project Structure

- `elicitation.py`: Contains functions to elicit information from users.
- `modeling.py`: Defines the `Model` class and methods to manipulate and update the model.
- `prompts.py`: Contains prompt templates for different stages of the elicitation process.
- `utils.py`: Utility functions for interacting with the OpenAI API.
- `demonstration.ipynb`: Jupyter Notebook demonstrating the usage of the elicitation and modeling functions.
- `.env`: Environment file for storing API keys and other configurations.
- `.gitattributes`: Git attributes configuration.
- `.gitignore`: Git ignore file to exclude unnecessary files from version control.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/khang200923/elicitation-and-modeling/
    cd elicitation-and-modeling
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up your OpenAI API key in the .env file:
    ```
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

You can run the demonstration notebook to see how the elicitation and modeling functions work together:

1. Start Jupyter Notebook:
    ```sh
    jupyter notebook
    ```

2. Open and run [[demonstration.ipynb]].

## License

This project is licensed under the MIT License.
