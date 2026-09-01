import glob
import os
import pickle
from tqdm import tqdm

def read_langevin_results(folder_path):

    pickle_files = glob.glob(os.path.join(folder_path, '*.pickle'))
    langevin_results = dict()
    
    for pickle_file in pickle_files:
        value_name = pickle_file.split(os.sep)[-1].split('.')[0]
        if value_name == 'samples':
            langevin_results['samples'] = read_pickle_in_chunks(pickle_file)
        else:
            with open(pickle_file, 'rb') as f:
                langevin_results[value_name] = pickle.load(f)
            

    return langevin_results

def read_pickle_in_chunks(file_path, chunk_size=1024):
    file_size = os.path.getsize(file_path)
    progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading pickle file")

    buffer = bytearray()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer.extend(chunk)
            progress_bar.update(len(chunk))

    progress_bar.close()
    data = pickle.loads(buffer)
    return data