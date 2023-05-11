import soundata

data_path = 'FSD50K/'
dataset = soundata.initialize('fsd50k', data_home=data_path)
dataset.download(cleanup=True)