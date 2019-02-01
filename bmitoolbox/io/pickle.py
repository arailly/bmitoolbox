import pickle

def load_pickle(file):
    with open(file, mode='rb') as f:
        return pickle.load(f)


def write_pickle(data, file):
    if file[-7:] != '.pickle':
        file += '.pickle'
    with open(file, mode='wb') as f:
        pickle.dump(data, f)
