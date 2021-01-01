import torchaudio
from glob import glob
from multiprocessing import Pool

rate = 44100
names = [file.split('/')[1][0:3] for file in sorted(glob('audio/*'))]


def f(name):
    waveform_, sample_rate = torchaudio.load('audio/'+name+'.mp3')
    #resample waveform to rate
    if sample_rate != rate:
        print(name,'need pocess')
        waveform_ = torchaudio.compliance.kaldi.resample_waveform(
                                            waveform=waveform_,
                                            orig_freq=sample_rate,
                                            new_freq=rate)
        torchaudio.save('audio/'+name+'.mp3',waveform_,sample_rate=rate)
        print(name,'done')

file = [name for name in names]
if __name__ == '__main__':  
    with Pool(35) as pool:  
        result = pool.map(f,file)