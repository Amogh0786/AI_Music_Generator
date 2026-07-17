import os
import pickle
import numpy as np
from music21 import corpus, converter, note, chord, instrument

def get_notes():
    print("Loading Bach chorales from music21 corpus...")
    # Load a smaller subset of bach chorales for faster training in this example
    paths = corpus.getComposer('bach')
    # Limit to 20 songs to keep preprocessing fast
    paths = paths[:20]
    
    notes = []
    
    for i, path in enumerate(paths):
        print(f"Parsing song {i+1} / {len(paths)}: {path}")
        midi = corpus.parse(path)
        
        notes_to_parse = None
        
        parts = instrument.partitionByInstrument(midi)
        
        if parts: # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
            
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                
    return notes

def prepare_sequences(notes, n_vocab):
    sequence_length = 50 # length of input sequence
    
    # get all pitch names
    pitches = sorted(set(item for item in notes))
    
    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitches))
    
    network_input = []
    network_output = []
    
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
        
    n_patterns = len(network_input)
    
    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length))
    network_output = np.array(network_output)
    
    return (network_input, network_output, note_to_int, pitches)

if __name__ == '__main__':
    notes = get_notes()
    n_vocab = len(set(notes))
    print(f"Total notes extracted: {len(notes)}")
    print(f"Vocabulary size: {n_vocab}")
    
    network_input, network_output, note_to_int, pitches = prepare_sequences(notes, n_vocab)
    print(f"Network input shape: {network_input.shape}")
    print(f"Network output shape: {network_output.shape}")
    
    with open('data.pkl', 'wb') as filepath:
        pickle.dump({
            'network_input': network_input,
            'network_output': network_output,
            'note_to_int': note_to_int,
            'pitches': pitches,
            'n_vocab': n_vocab
        }, filepath)
    
    print("Saved preprocessed data to data.pkl")
