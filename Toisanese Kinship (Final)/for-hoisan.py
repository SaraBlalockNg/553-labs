from scipy.interpolate import spline
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, re
from os.path import join
from praatio import tgio
from scipy import stats

minpitch = 40
maxpitch = 200

intensity_min = 45 # change me!
intensity_max = 140 # change me!

### change all these for tsudoi ###
output_path = r"C:\Users\Sara\Desktop\for553"
praatEXE = r"C:\Users\Sara\Desktop\Praat.exe"
end1= r"Outputs\PI"
end2= r"Outputs\Splits"
###################################

PIoutputpath = join(output_path,end1)
splitpath = join(output_path,end2)
wavpath = join(output_path,'Concatenated.wav') # change me!
tgpath = wavpath[:-3]+'textGrid'

def read_listing_files():    
    tg = tgio.openTextgrid(tgpath)

    duration = dict()
    duration_f = dict()
    words_f = []
    words = []
    # change this tiername for tsudoi

    for start,stop,_ in tg.tierDict['voiced_f'].entryList:
        words_f.append(_)
    for start,stop,_ in tg.tierDict['voiced_nf'].entryList:
        words.append(_)
    for start,stop,_ in tg.tierDict['morpheme_nf'].entryList:
        try:
            duration[_].append((stop-start)*1000)
        except KeyError:
            duration[_]=[(stop-start)*1000]
    # change this tiername for tsudoi
    for start,stop,_ in tg.tierDict['morpheme_f'].entryList:
        try:
            duration_f[_].append((stop-start)*1000)
        except KeyError:
            duration_f[_]=[(stop-start)*1000]
    return(duration,duration_f,words,words_f)

def extract(path):
    with open(path,'rb') as f:
        g= f.read().decode()
    g = re.sub('--undefined--','4041215665',g)
    g = g.split('\r\n')[1:]
    time= [float(a.split()[0]) for a in g if len(a)>0]
    val = [float(a.split()[1]) for a in g if len(a)>0]
    return(time,val)

def make_info(words,words_f):
    all_words = sorted(list(set(words+words_f)))

    ### change these for tsudoi ###
    thedir = r"C:\Users\Sara\Desktop\for553\Outputs\f"
    theotherdir = r"C:\Users\Sara\Desktop\for553\Outputs"
    ###############################
    
    d = dict()
    for w in all_words:
        d[w]=([],[],[],[],[],[],[],[])
    
    # each file should have a p and i file
    # change me! should be the # in the file names
    for i in range(3,48):
        try:
            res1,res2 = extract(join(theotherdir,'p'+str(i)))
            res3,res4 = extract(join(theotherdir,'i'+str(i)))
            d[words_f[i-3]][1].append(res1)
            d[words_f[i-3]][5].append(res2)
            d[words_f[i-3]][3].append(res3)
            d[words_f[i-3]][7].append(res4)
        except IndexError:
            print(i)
            print(len(words_f))
    # change me! should be the number in the file names
    for i in range(3,147): 
        res1,res2 = extract(join(thedir,'p'+str(i)))
        res3,res4 = extract(join(thedir,'i'+str(i)))
        d[words[i-3]][2].append(res1)
        d[words[i-3]][6].append(res2)
        d[words[i-3]][0].append(res3)
        d[words[i-3]][4].append(res4)
    return(d)

def sliceit (li,l):
    start = 0
    new = []
    for i in li:
        stop = start+i
        new.append(l[start:stop])
        start = stop
    return(new)

def store_info(word,lists):
    scaler = MinMaxScaler(feature_range=(0,1))

    # first, scale the times
    f_int_time = [scaler.fit_transform(np.float32(a)[:,np.newaxis]).T[0]
                  for a in lists[0]]
    int_time = [scaler.fit_transform(np.float32(a)[:,np.newaxis]).T[0]
                for a in lists[1]]
    f_pitch_time = [scaler.fit_transform(np.float32(a)[:,np.newaxis]).T[0]
                    for a in lists[2]]
    pitch_time = [scaler.fit_transform(np.float32(a)[:,np.newaxis]).T[0]
                  for a in lists[3]]
    
    # then, flatten the lists
    f_int_time = [item for sublist in f_int_time for item in sublist]
    int_time = [item for sublist in int_time for item in sublist]
    f_pitch_time = [item for sublist in f_pitch_time for item in sublist]
    pitch_time = [item for sublist in pitch_time for item in sublist]
    l1 = [len(a) for a in lists[4]]
    l2 = [len(a) for a in lists[5]]
    l3 = [len(a) for a in lists[6]]
    l4 = [len(a) for a in lists[7]]
    l5 = [len(a) for a in lists[0]]
    l6 = [len(a) for a in lists[1]]
    l7 = [len(a) for a in lists[2]]
    l8 = [len(a) for a in lists[3]]

    f_int_values = list(stats.zscore([item for sublist in lists[4] for item in sublist]))
    int_values = list(stats.zscore([item for sublist in lists[5] for item in sublist]))
    f_pitch_values = list(stats.zscore([item for sublist in lists[6] for item in sublist]))
    pitch_values = list(stats.zscore([item for sublist in lists[7] for item in sublist]))

    fit = []
    it = []
    fpt = []
    pt = []
    fiv = []
    iv = []
    fpv = []
    pv = []
    for i in range(len(f_int_time)):
        if not f_int_values[i]==4041215665:
            fit.append(f_int_time[i])
            fiv.append(f_int_values[i])
    for i in range(len(int_time)):
        if not int_values[i]==4041215665:
            it.append(int_time[i])
            iv.append(int_values[i])
    for i in range(len(f_pitch_time)):
        if not f_pitch_values[i]==4041215665:
            fpt.append(f_pitch_time[i])
            fpv.append(f_pitch_values[i])
    for i in range(len(pitch_time)):
        if not pitch_values[i]==4041215665:
            pt.append(pitch_time[i])
            pv.append(pitch_values[i])

    fiv = sliceit(l1,fiv)
    iv= sliceit(l2,iv)
    fpv = sliceit(l3,fpv)
    pv = sliceit(l4,pv)
    fit = sliceit(l5,fit)
    it = sliceit(l6,it)
    fpt = sliceit(l7,fpt)
    pt = sliceit(l8,pt)

    all_data_for_word = (fit,fiv,it,iv,fpt,fpv,pt,pv)

    return(all_data_for_word)

def plot_tone(word,fpt,fpv,pt,pv):
    fig, ax = plt.subplots()
    for i in range(len(fpt)):
        x1 = np.array(fpt[i])
        x1_new = np.linspace(x1.min(),x1.max(),300)
        y1 = np.array(fpv[i])
        try:
            smooth_y1 = spline(x1,y1,x1_new)
            if i == 0:
                # change label for tsudoi
                ax.plot(x1_new,smooth_y1,color='m',label=str(word)+' non-focused',linewidth=3.0)
            else:
                ax.plot(x1_new,smooth_y1,color='m',linewidth=3.0)
        except ValueError:
            pass
    for i in range(len(pt)):
        x2 = np.array(pt[i])   
        x2_new = np.linspace(x2.min(),x2.max(),300)
        y2 = np.array(pv[i])
        try:
            smooth_y2 = spline(x2,y2,x2_new)
            if i == 0:
                # change for tsudoi
                ax.plot(x2_new,smooth_y2,color='c',label=str(word)+' contrast-focused',linewidth=3.0)
            else:
                ax.plot(x2_new,smooth_y2,color='c',linewidth=3.0)
        except ValueError:
            pass

    ### change for tsudoi ####
    plt.xlabel('normalized time')
    plt.ylabel('z-score of pitch')
    plt.legend(loc=3)
    plt.ylim((-5,5))
    plt.title('Pitch Contour')

    plt.savefig('C:/Users/Sara/Desktop/for553/pitch_for_'+str(word)+'.png')
    plt.close()
    ###########################

def plot_intensity(word,fit,it,fiv,iv):
    fig, ax = plt.subplots()
    for i in range(len(fit)):
        x1 = np.array(fit[i])
        x1_new = np.linspace(x1.min(),x1.max(),300)
        y1 = np.array(fiv[i])
        try:
            smooth_y1 = spline(x1,y1,x1_new)
            if i == 0:
                ax.plot(x1_new,smooth_y1,color='m',label=str(word)+' non-focused',linewidth=3.0)
            else:
                ax.plot(x1_new,smooth_y1,color='m',linewidth=3.0)
        except ValueError:
            pass
    for i in range(len(it)):
        x2 = np.array(it[i])   
        x2_new = np.linspace(x2.min(),x2.max(),300)
        y2 = np.array(iv[i])
        try:
            smooth_y2 = spline(x2,y2,x2_new)
            if i == 0:
                ax.plot(x2_new,smooth_y2,color='c',label=str(word)+' contrast-focused',linewidth=3.0)
            else:
                ax.plot(x2_new,smooth_y2,color='c',linewidth=3.0)
        except ValueError:
            pass

    #### change for tsudoi ###
    plt.xlabel('normalized time')
    plt.ylabel('z-score of intensity')
    plt.legend(loc=3)
    plt.title('Intensity Contour')
    plt.ylim((-5,5))
    plt.savefig('C:/Users/Sara/Desktop/for553/intensity_for_'+str(word)+'.png')
    plt.close()
    ##########################

def plot_duration(d):
    i = 0
    names = []
    values = []
    for k in sorted(d.keys()):
        v = d[k]
        for value in v:
            names.append(k)
            values.append(value)
        i+=1    
    h = sorted(list(d.keys()))
    df = {'Speech Type':names,'Duration (in ms)':values}
    df = pd.DataFrame(data=df)
    x = df.boxplot('Duration (in ms)',by='Speech Type',figsize=(8,4),fontsize=6)
    x.get_figure().suptitle('')
    x.get_figure().gca().set_title('Duration for each morpheme type')
    x.get_figure().gca().set_xlabel('Speech Type')
    x.get_figure().gca().set_ylabel('Time (in ms)')
    plt.savefig('C:/Users/Sara/Desktop/for553/duration.png')
    
def main():
    d,df,words,words_f = read_listing_files()
    for key in df.keys():
        d[key+'_f']=df[key]
    all_words = list(set(words+words_f))
    dict_of_values = make_info(words,words_f)
    user_dict = dict()
    for word in all_words:
        user_dict[word] = store_info(word,dict_of_values[word])
    for word,value in user_dict.items():
        plot_intensity(word,value[0],value[2],value[1],value[3])
        plot_tone(word,value[4],value[5],value[6],value[7])
    plot_duration(d)

main()
