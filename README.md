# GPT-J Chatbot

This project demonstrates how to build, fine-tune, and deploy a chatbot using the GPT-J model.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/rick001/gpt-chatbot.git
    cd gpt-chatbot
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv gpt_env
    source gpt_env/bin/activate
    ```

3. Install the dependencies:
    ```bash
    pip install -r app/requirements.txt
    ```

## Fine-Tuning

To fine-tune the model on your dataset:
1. Place your training data in `data/training_data.txt`.
2. Run the fine-tuning script:
    ```bash
    python app/fine_tune.py
    ```

## Running the Chatbot

To start the Flask application:
```bash
python app/chatbot.py
