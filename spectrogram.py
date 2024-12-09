import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import os

""" short time fourier transform of audio signal """



def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
# binsize normally 2**10 I like 128
def plotstft(audiopath, binsize=128, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)

    #print(len(samples))

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    #plt.colorbar()

    #plt.xlabel("time (s)")
    #plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
    
    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()
    #plt.close()

    return ims

'''
#directory = "/Users/nhogan/Desktop/SpectrogramTest/soerenab AudioMNIST master data-01"
#directory = "/Users/nhogan/Desktop/SpectrogramTest/soerenab AudioMNIST master data-02"
directory = "/Users/nhogan/Desktop/SpectrogramTest/z" # change this

for filename in os.scandir(directory):
    if filename.is_file():
        ims = plotstft(filename, plotpath="zSpec/"+str(filename)+".png") # change this
'''

directory = "/Users/nhogan/Desktop/SpectrogramTest/temp_wavs" # change this

for filename in os.scandir(directory):
    if filename.is_file():
        print(str(filename))
        if(str(filename)[-3] == 'v'):
            ims = plotstft(filename, plotpath="/Users/nhogan/Desktop/Comps??/data/spectrograms/train_new/ash/"+str(filename)+".png") # change this

#ims = plotstft("0_01_0.wav")
#ims = plotstft("common_voice_es_18310029-2_new.Sound")


#ims = plotstft("/Users/nhogan/Desktop/languages/en/grid_and_wav/common_voice_en_110270.wav", plotpath="110270.png")
