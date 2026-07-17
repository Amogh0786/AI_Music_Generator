# AI Music Generator

This project generates music using an LSTM neural network. It trains on MIDI data (Bach chorales by default) and generates new sequences of notes, which are then saved as MIDI files.

## Project Structure

- `preprocess.py`: Loads the MIDI data, extracts notes and chords, maps them to integers, and prepares sequences for training.
- `model.py`: Defines the PyTorch Long Short-Term Memory (LSTM) network architecture.
- `train.py`: Trains the model on the preprocessed sequences and saves the trained weights to `music_model.pth`.
- `generate.py`: Loads the trained model, generates a new sequence of notes, and saves it as `generated_music.mid`.
- `requirements.txt`: Contains the required Python libraries.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Preprocess Data**:
   ```bash
   python preprocess.py
   ```
   *This will generate `data.pkl` containing the processed sequences and vocabulary.*

2. **Train the Model**:
   ```bash
   python train.py
   ```
   *This will train the LSTM network and save the weights to `music_model.pth`.*

3. **Generate Music**:
   ```bash
   python generate.py
   ```
   *This will use the trained model to generate a new music sequence and save it as `generated_music.mid`.*
