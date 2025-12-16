gdown --fuzzy https://drive.google.com/file/d/1v4aDGZsXmNsFAxQ9D7IoPes-pko7QRzc/view?usp=share_link -O data.json
data = pd.read_json('/content/data.json')
data = pd.json_normalize(data['data'])
data = data.dropna(axis=1, how="all")