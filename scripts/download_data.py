from pathlib import Path
import zipfile,urllib.request
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'data/raw'; out.mkdir(parents=True,exist_ok=True)
url='https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'; z=out/'ml-latest-small.zip'
if not z.exists(): print('Downloading MovieLens...'); urllib.request.urlretrieve(url,z)
with zipfile.ZipFile(z) as f:f.extractall(out)
print('Dataset ready:',out/'ml-latest-small')
