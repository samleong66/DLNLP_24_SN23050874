import os
import requests, zipfile
from tqdm import tqdm

def get_parser():
    if not os.path.exists("B\stanford parser\stanford-parser.jar"):
        print('downloading stanford parser...')
        zip_file = 'B/stanford parser.zip'
        url = 'https://nlp.stanford.edu/software/stanford-parser-full-2015-12-09.zip'
        response = requests.get(url, stream=True)
        pbar = tqdm(total=int(response.headers.get('Content-Length')))
        with open(zip_file, 'wb') as f:
            for chunk in response.iter_content(1024*1024):
                f.write(chunk)
                pbar.set_description("Downloading")
                pbar.update(1024*1024)
        pbar.close()
        file = zipfile.ZipFile(zip_file)
        file.extractall('B/')
        file.close()
        os.remove(zip_file)
        os.rename('B/stanford-parser-full-2015-12-09', 'B/stanford parser')
    else:
        print('Stanford parser installed!')
