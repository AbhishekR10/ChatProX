# Basic ChatBot

## Description:

The Chatbot Project is a machine learning-based chatbot developed using Python and TensorFlow/Keras for natural language processing.

## Intents Definition:

Intents are defined in intents.json, which includes tags, patterns (input messages), and corresponding responses.
json file:

{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hey", "Is anyone there?", "Hello", "Hay"],
      "responses": ["Hello", "Hi", "Hi there"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye"],
      "responses": ["See you later", "Have a nice day", "Bye! Come back again"]
    },
    ...
  ]
}

## Model Training:

The chatbot model is trained using TensorFlow/Keras. It utilizes an embedding layer followed by dense layers to classify intents based on input messages.
Tokenizer: Converts text to sequences and pads them to a maximum length.
Model: Sequential model with Embedding, GlobalAveragePooling1D, and Dense layers.
Training: Trained on the provided intents data.
train_model.py: Script to train the chatbot model.
chatbot.py: Script to interact with the chatbot.
Training: Run train_model.py to train the chatbot model.
Chatting: Execute chatbot.py to interact with the chatbot.

## Requirements:

Python 3.x
TensorFlow
NumPy
Pandas
Scikit-Learn

# Advanced ChatBot

## Description:

Welcome to AdvancedChatBot, a smart and engaging chatbot designed to provide various functionalities, including conversation, information retrieval, and more. This project leverages natural language processing to create an interactive and user-friendly chatbot experience.

## Dataset:

The dataset consists of transcripts from the animated television show "Rick and Morty," specifically from Season 1, Episode 11. These transcripts capture dialogues between various characters, including Rick, Morty, Jerry, Beth, Summer, and several other characters. The dataset is intended for use in natural language processing tasks such as text analysis, sentiment analysis, and chatbot training.
## Dataset Structure:
The dataset is organized as follows:
Transcripts: The main content of the dataset, capturing the spoken dialogue from the episode.
Metadata: Information about the episode, such as season and episode numbers, character names, and context of the dialogue.
## File Structure:
The dataset file contains the following columns:
Column Name	Description:
'Speaker'--The character who is speaking
'Dialogue'--The text of the dialogue spoken by the character
'Show'--The name of the show (Rick and Morty)
'Season'--The season number of the episode
'Episode'--The episode number within the season
## Data Fields:
Example Value"
Speaker	Rick
'Dialogue'--Wubbalubbadubdub! Oh. Hey Jerry. Wh-wh-what are you doing in my room, buddy?
'Context'--Article stubs, Rick and Morty Transcripts
'Show'--Rick and Morty(Season 1 Episode 11)

## Discord Bot Instance Details:

This Python script defines a Discord bot (MyClient) that interacts with a Hugging Face model (DialoGPT-medium-joshua) through its API. The bot uses the discord library to connect to Discord servers and requests to make HTTP requests to the model API. It listens for messages in Discord channels, sends them to the model for processing, and posts the model's responses back to the channel. The bot is initialized with environment variables (HUGGINGFACE_TOKEN for the model API and DISCORD_TOKEN for Discord bot authentication). It demonstrates basic usage of asynchronous event handling (on_ready and on_message methods) and error handling for API responses.

## Model Training Details:

## 1. Setup and Dependencies:
Mounted Google Drive to Colab.
Installed necessary libraries like transformers.
Imported required modules and packages.

## 2. Data Acquisition and Preparation:
Downloaded dataset from Kaggle using kaggle.json.
Preprocessed the data to create conversational context windows (contexted) around responses from the character "Joshua".

## 3. Dataset Creation:
Constructed a custom dataset (ConversationDataset) suitable for training the model, ensuring proper tokenization and padding.

## 4. Model Training:
Defined training functions (train) to handle distributed training, gradient accumulation, and checkpointing.
Implemented evaluation functions (evaluate) to assess model performance during and after training.

## 5. Model Configuration and Initialization:
Utilized AutoModelWithLMHead and AutoTokenizer to load and configure the DialoGPT model.
Configured training parameters (Args class) including batch sizes, learning rate, and epochs.

## 6. Training Execution:
Ran the main training function (main) to train the model using the prepared dataset and specified configurations.
Saved the trained model, tokenizer, and training arguments for future use.

## 7. Model Deployment:
Loaded the trained model and tokenizer for inference.
Interacted with the model by generating responses based on user input.

## 8. Push to Hugging Face Model Hub:
Configured Git and installed Git LFS for pushing the trained model and tokenizer to the Hugging Face Model Hub.
