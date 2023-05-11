import soundata

data_path = 'US8k/'
dataset = soundata.initialize('urbansound8k', data_home=data_path)
dataset.download(cleanup=True)