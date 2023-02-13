# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 23:27:02 2023

@author: pballing
"""

import pandas as pd
import json
from datetime import datetime
from datetime import timedelta
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import re
import math
import numpy as np
import time
from sys import getsizeof
import sqlite3
from scipy.fftpack import fft, ifft
from scipy.signal import butter,filtfilt

def set_up_axes(ax,title,ylabel,time_delta):
    print(time_delta)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.margins(x=0)
    ax.set_xticks(mdates.MinuteLocator(interval=int(time_delta["interval"]/8)).tick_values(time_delta["start"], time_delta["end"]))
    ax.set_xticklabels(ax.get_xticks(), rotation = 45)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=int(time_delta["interval"]/8)))
    #ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=time_delta["interval"]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    
    if ylabel =="Mbps":
        ax.set_yticks(range(0,120,10))
    
def output_to_javascript(summed_df):
    summed_df.reset_index(inplace=True)
    summed_df = summed_df.rename(columns = {'index':'time'})
    #print(summed_df)
    summed_df["time"] = summed_df["time"].dt.strftime('%Y-%m-%d %H:%M:%S')
 
    summed_df["port"] = "Port_1"
    data = summed_df.to_dict('records')
    """with open("static/data.json","w") as f:
        r = json.dumps(data)
        f.write(r)"""
    return data
    #print(mylist[0:10])
def raw_data_to_pandas(filenameR,filenameW):
    data = []
    df = pd.DataFrame()
    with open(filenameR,'r') as f:
        for line in f:
            line = json.loads(line)
            data.append(line)
    #print(getsizeof(data))
    df = pd.DataFrame(data)
    #print(getsizeof(df))
    df["time"] = pd.to_datetime(df["time"], format='%Y-%m-%d %H:%M:%S')
    #race condition, loop ran twice but reported at the same second
    print(f"duplicate times for different values: {df[df.duplicated(subset=['time','port'],keep=False)]}")
    df.drop_duplicates(subset=['time','port'], inplace=True)
    df = df.set_index("time")
    for col in df.columns:
        if col != "port":
            df[col] = df[col].astype(float)
    
    #print("data loaded")
    #print(getsizeof(df))
    #Get rid of null rows in the data, {} in the raw data from the file
    df.dropna(axis=0,inplace=True)
    return df
def create_SQLite_table(database,df,raw_data_file):
    #print(df)
    conn = sqlite3.connect(database)
    c = conn.cursor()
    #c.execute('CREATE TABLE IF NOT EXISTS data (time TIMESTAMP, port text, bitstx REAL, bitsrx REAL, pkttx REAL, pktrx REAL)')
    #use double quotes to escape the periods in the IP
    c.execute('CREATE TABLE IF NOT EXISTS "' + raw_data_file + '" (time TIMESTAMP NOT NULL, port CHARACTER(20) NOT NULL)')
    c.execute('SELECT * FROM pragma_table_info("' + raw_data_file + '")')
    rows = c.fetchall()
    column_names = []
    for entry in rows:
        column_names.append(entry[1])
    print(column_names)
    for column in df.columns:
        if column not in column_names:  #see if column names exist
            c.execute('ALTER TABLE "' + raw_data_file + '" ADD COLUMN ' + column + ' REAL')
    conn.commit()
    
    df.to_sql(raw_data_file, conn, if_exists='append', index = True)

    conn.close()
def read_from_SQL(database,raw_data_file):
    #conn = sqlite3.connect(database,detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) 
    conn = sqlite3.connect(database)
    #c = conn.cursor()
    df = pd.read_sql('SELECT * FROM "' + raw_data_file + '";',conn)
    df["time"] = pd.to_datetime(df["time"], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index("time")

    return df
def read_from_SQL1(database):
    #Slower than read_from_SQL()
    conn = sqlite3.connect(database,detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) 
    #conn = sqlite3.connect(database)
    #c = conn.cursor()
    df = pd.read_sql("SELECT * FROM data",conn)
    df = df.set_index("time")

    return df
def analyze_df(df):
    search_data_keysTx = re.compile("(\S+)[Tt]x")
    search_data_keysRx = re.compile("(\S+)[Rr]x")
    data_keysTx = []
    data_keysRx = []
    matched_Tx_Rx_keys = []
    ports = df.port.unique().tolist()
    print(ports)
    for col in df.columns:
        m = search_data_keysTx.search(col)
        if m:
            data_keysTx.append(m[1])
        m = search_data_keysRx.search(col)
        if m:
            data_keysRx.append(m[1])
    for data_key in data_keysTx:
        if data_key in data_keysRx:
            matched_Tx_Rx_keys.append(data_key)
    print(matched_Tx_Rx_keys)
    return ports,matched_Tx_Rx_keys
def time_slices(temp_df):
    time_delta={"start":temp_df.first_valid_index(),"end":temp_df.last_valid_index()}
    myrange = time_delta["end"] - time_delta["start"]
    #print(myrange.total_seconds())
    time_delta.update({"interval":int(int(myrange.total_seconds())/720)})   #about 12 ticks on x axis finding minute locator, 12 * 60 seconds
    if time_delta["interval"] == 0:
        time_delta["interval"] = 1
    return time_delta
def simple_plot_full(df,units_for_data_key,savefile):
    df = df.tail(86400)
    ports,matched_Tx_Rx_keys = analyze_df(df)
    f, axes = plt.subplots(len(ports), len(matched_Tx_Rx_keys), figsize=(480,32))
    for port_index,port in enumerate(ports):
        temp_df = df[df["port"] == port]
        for key_index,data_key in enumerate(matched_Tx_Rx_keys):
            if len(ports) ==1:
                if len(matched_Tx_Rx_keys) == 1:
                    ax = temp_df.plot(y=[data_key+"tx",data_key+"rx"],ax=axes)
                else:
                    ax = temp_df.plot(y=[data_key+"tx",data_key+"rx"],ax=axes[key_index])
            elif len(matched_Tx_Rx_keys) == 1:
                ax = temp_df.plot(y=[data_key+"tx",data_key+"rx"],ax=axes[port_index])
            else:
                ax = temp_df.plot(y=[data_key+"tx",data_key+"rx"],ax=axes[port_index][key_index])
            time_delta=time_slices(temp_df)
            print(temp_df)
            set_up_axes(ax,port + ' '+data_key,units_for_data_key[data_key],time_delta)
    print("plot prepped")
    #plt.tight_layout()
    plt.show()
    print("saving to file")
    if len(ports) ==1:
        if len(matched_Tx_Rx_keys) == 1:
            axes.get_figure().savefig(savefile)
        else:
            axes[0].get_figure().savefig(savefile)
    elif len(matched_Tx_Rx_keys) == 1:
        axes[0].get_figure().savefig(savefile)
    else:
        axes[0][0].get_figure().savefig(savefile)

def summarize_to_fewer_points_MAX(df,periods = 0, freq =0):
    df = df[df["port"] == "1"]
    if freq == 0 and periods == 0:
        my_date_range =pd.date_range(start=df.first_valid_index(),end=df.last_valid_index(),periods = 2000)
    elif freq == 0:
        my_date_range =pd.date_range(start=df.first_valid_index(),end=df.last_valid_index(),periods = periods)
    elif periods == 0:
        my_date_range =pd.date_range(start=df.first_valid_index(),end=df.last_valid_index(),freq =str(freq) + 'S')
    mylist = []
    for i in range(len(my_date_range)-1):
        mylist.append(df.loc[my_date_range[i]:my_date_range[i+1]].max())
        #summed_df = pd.concat([summed_df,{"time":my_date_range[i+1],"bitstx":max_of_range}],ignore_index=True)
    my_date_range = my_date_range.delete(0)
    summed_df = pd.DataFrame(mylist,index=my_date_range)
    #print(summed_df)
    return summed_df
def summarize_to_2000_points_MAX(df, **kwargs):
    df = df[df["port"] == "1"]
    if "periods" not in kwargs:
        periods = 2000
    else:
        periods = kwargs["periods"]
    if "my_date_range" not in kwargs:
        my_date_range =pd.date_range(start=df.first_valid_index(),end=df.last_valid_index(),periods = periods)
    else: 
        print(kwargs["my_date_range"])
        start = pd.to_datetime(kwargs["my_date_range"][0], format='%Y-%m-%d %H:%M:%S')
        end = pd.to_datetime(kwargs["my_date_range"][1], format='%Y-%m-%d %H:%M:%S')
        print(start)
        print(end)
        my_date_range =pd.date_range(start=start,end=end,periods = periods)
    mylist = []
    for i in range(len(my_date_range)-1):
        mylist.append(df.loc[my_date_range[i]:my_date_range[i+1]].max())
        #summed_df = pd.concat([summed_df,{"time":my_date_range[i+1],"bitstx":max_of_range}],ignore_index=True)
    #print(my_date_range)
    my_date_range = my_date_range.delete(0)
    summed_df = pd.DataFrame(mylist,index=my_date_range)
    #print(summed_df)
    return summed_df
def summarize_to_2000_points_MEAN(df):
    df = df[df["port"] == "1"]

    my_date_range =pd.date_range(start=df.first_valid_index(),end=df.last_valid_index(),periods = 2000)
    mylist = []
    for i in range(len(my_date_range)-1):
        mylist.append(df.loc[my_date_range[i]:my_date_range[i+1]].mean())
        #summed_df = pd.concat([summed_df,{"time":my_date_range[i+1],"bitstx":max_of_range}],ignore_index=True)
    my_date_range = my_date_range.delete(0)
    summed_df = pd.DataFrame(mylist,index=my_date_range)
    
    summed_df["port"] = "1"
    #print(summed_df)
    return summed_df
def summarize_to_2000_points_ROLLING(df):
    df = df[df["port"] == "1"]

    summed_df = df.rolling(60*5).mean()
    summed_df["port"] = "1"
    print(summed_df)
    return summed_df
def create_histogram_full(df,clip,bins):
    df_filtered = df[df["port"] == "1"]
    df_filtered.loc["bitstx"] = df_filtered["bitstx"].clip(upper=clip,inplace = True)
    f, axes = plt.subplots(4, 1, figsize=(16,16))
    df_filtered.hist(ax=axes[0],column='bitstx',bins=bins)
    df_filtered[df_filtered["bitstx"] > 10].hist(ax=axes[1],column='bitstx',bins=bins)
    df_filtered[df_filtered["bitstx"] > 20].hist(ax=axes[2],column='bitstx',bins=bins)
    df_filtered[df_filtered["bitstx"] > 30].hist(ax=axes[3],column='bitstx',bins=bins)
    
    plt.show()
def FFT(df,sr,ax):
    df = df[df["port"] == "1"]
    print(df['bitstx'])
    X = fft(df['bitstx'].values )
    N = len(X)
    print(f'N: {N}')
    n = np.arange(N)
    print(f'n: {n}')
    # get the sampling rate
    #sr = 1 / (60*60)
    sr = 1 / sr
    T = N/sr
    freq = n/T 
    
    # Get the one-sided specturm
    n_oneside = N//2
    # get the one side frequency
    f_oneside = freq[:n_oneside]
    
    #plt.figure(figsize = (12, 6))
    ax.plot(f_oneside, np.abs(X[:n_oneside]), 'b')
    ax.set_xlabel('Freq (Hz)')
    ax.set_ylabel('FFT Amplitude |X(freq)|')
    ax.set_xlim(-.00001,1/(60*60*12))
    #ax.show()
    return X
def FFT_hour(df,sr,ax):
    df = df[df["port"] == "1"]
    #print(df['bitstx'])
    X = fft(df['bitstx'].values)
    N = len(X)
    
    n = np.arange(N)
    # get the sampling rate
    #sr = 1 / (60*60)
    sr = 1 / sr
    T = N/sr
    freq = n/T 
    
    # Get the one-sided specturm
    n_oneside = N//2
    # get the one side frequency
    f_oneside = freq[:n_oneside]
    
    # convert frequency to hour
    t_h = 1/f_oneside / (60 * 60)
    
    #plt.figure(figsize=(12,6))
    ax.plot(t_h, np.abs(X[:n_oneside])/n_oneside)
    #ax.set_xticks([0,1,2,3,6,12, 24, 36,48,60,72,84,96,167])
    ax.set_xticks([.25,.5,.75,1,2])
    ax.set_title("FFT no filtering");
    ax.set_ylabel('Amplitude')
    ax.grid()
    ax.set_xlabel('Period ($hour$)')
    #plt.show()
    return X
def IFFT(X,ax):
    #print(X)
    newgraph = ifft(X)
    #print(newgraph)
    ax.plot(newgraph)
    ax.set_title("original graph plotted using inverse Fourier transform");
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.grid()
    #plt.show();
    return newgraph
def butter_lowpass_filter(df,ax):
    df = df[df["port"] == "1"]
    
    # Filter requirements.
    T = 662262         # Sample Period
    fs = 1       # sample rate, Hz
    cutoff = 1/(60*60*24)    # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2       # sin wave can be approx represented as quadratic
    n = int(T * fs) # total number of samples
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, df['bitstx'].values )
    #f, axes = plt.subplots(1, 1, figsize=(32,32))
    ax.plot(y)
    ax.set_xlabel('time')
    ax.set_ylabel('Mbps')
    ax.set_title('Low Pass')
    ax.grid()
    newdf = pd.DataFrame({"bitstx": y.astype(float),"port":"1"})
    return newdf

def Plot_FFTs(df,summed_df):
    df = df[df["port"] == "1"]
    #plt.figure(figsize = (12, 6))
    no_time = pd.DataFrame(df['bitstx'].values,columns=["raw"])
    print(no_time)
    f, axes = plt.subplots(4, 2, figsize=(32,32))
    axes[0][0].plot(df['bitstx'].values)
    axes[0][0].set_xlabel('time')
    axes[0][0].set_ylabel('Mbps')
    axes[0][0].set_title('raw')
    axes[0][0].grid()
    #X = FFT_hour(df,total_seconds/4000,axes[0][1])
    X = FFT_hour(df,1,axes[0][1])
    newdf = butter_lowpass_filter(df,axes[1][0])
    print(newdf)
    FFT_hour(newdf,1,axes[1][1])
    newgraph = IFFT(X,axes[2][0])
    axes[3][1].plot(summed_df['bitstx'].values)
    axes[3][1].set_xlabel('time')
    axes[3][1].set_ylabel('Mbps')
    axes[3][1].set_title('Mean value over time slice')
    axes[3][1].grid()
    print(len(newgraph))
    no_time_fromifft = pd.DataFrame(newgraph.astype(float))
    no_time["ifft"] = no_time_fromifft
    no_time["diff"] = np.where(no_time["raw"] == no_time["ifft"], 0 , (no_time["raw"] - no_time["ifft"]).round(3))
    print(no_time)
    axes[3][0].plot(no_time["diff"].values)
    axes[3][0].set_xlabel('time')
    axes[3][0].set_ylabel('Mbps')
    axes[3][0].set_title('diff')
    axes[3][0].grid()
    
    axes[2][1].plot(summed_df['diff'].values)
    axes[2][1].set_xlabel('time')
    axes[2][1].set_ylabel('Mbps')
    axes[2][1].set_title('diff between Mean value and next value')
    axes[2][1].grid()
    #axes[0][0]= df.plot()
    axes[0][0].get_figure().savefig("Filename.png")
def plot_adtran(df, **kwargs):
    
    #Get rid of null rows in the data, {} in the raw data from the file
    df.dropna(axis=0,inplace=True)
    ports,matched_Tx_Rx_keys = analyze_df(df)
    print(matched_Tx_Rx_keys)
    if "start1" not in kwargs:
        start1 = df.first_valid_index()
    else:
        start1 = kwargs["start1"]
    if "end1" not in kwargs:
        end1 = df.first_valid_index()
    else:
        end1 = kwargs["end1"]
    df.drop(["port"],axis=1,inplace=True)
    if "columns" in kwargs:
        columns = kwargs["columns"]
        for col in df.columns:
            if col not in columns:
                df.drop(col,axis=1,inplace=True)
    df = df.diff()
    start = df.first_valid_index()
    end = df.last_valid_index()

    df.reset_index(inplace=True)
    dates = pd.date_range(start=start, end=end, freq='s')
    df = df.set_index('time').reindex(dates)

    #fill individual seconds with the value over 6 seconds previously divided by 6
    #--->df.interpolate(method='backfill',limit_direction='backward', limit=5,inplace=True)

    #DAQ most likely will not report 1 second earlier than expected because async.sleep(5) + async.sleep(1)
    df = pd.concat([df,(df.Bytesrx.isnull().astype(int).groupby(df.Bytesrx.notnull().astype(int).cumsum()).cumsum().to_frame('consec_count'))],axis=1)
    df["count_diff"] = df['consec_count'].diff()
    df.dropna(axis=0,inplace=True)
    """print("Add count Nan columns")
    print(df.head(30)[["Bytesrx","count_diff","consec_count"]])"""
    
    for col in df.columns:
        if col != "count_diff" and col != "consec_count":
            df[col] = df[col] / ((df["count_diff"] * -1) + 1)
    
    """print("divide data over elasped seconds")
    print(df.head(30)[["Bytesrx","count_diff","consec_count"]])
    print(df)"""
    df.reset_index(inplace=True)
    df.rename(columns={"index": "time"},inplace=True)
    dates = pd.date_range(start=start, end=end, freq='s')
    df = df.set_index('time').reindex(dates) 
    df.fillna(method="backfill",inplace=True)
    """print("backfill")
    print(df.head(30)[["Bytesrx","count_diff","consec_count"]])
    print(df)
    print(df.loc[df["Broadcaststx"]==df["Broadcaststx"].max()]["Broadcaststx"])"""
    df["Bytesrx"] = (df["Bytesrx"] * 8)/1000000
    df["Bytestx"] = (df["Bytestx"] * 8)/1000000

    if "axes" not in kwargs:
        f, axes_all = plt.subplots(len(matched_Tx_Rx_keys), 1, figsize=(32,32))
    else: 
        axes_all = [kwargs["axes"]]
        f = kwargs["axes"].get_figure()
    #print(axes)
    for i,axes in enumerate(axes_all):
        #print(matched_Tx_Rx_keys[i])
        axes.plot(df[[matched_Tx_Rx_keys[i]+"rx",matched_Tx_Rx_keys[i]+"tx"]])
        #axes.plot(df.index,df['bitsrx'].values)
        axes.set_xlabel('time')
        if matched_Tx_Rx_keys[i] == "Bytes":
            axes.set_ylabel('Mbps')
        else:
            axes.set_ylabel('Packets per second')
        axes.set_title(matched_Tx_Rx_keys[i])
        #print(df.columns)
        axes.legend([matched_Tx_Rx_keys[i]+"rx",matched_Tx_Rx_keys[i]+"tx"],loc='upper left',prop={'size': 20})
        axes.grid()
        #start = df.first_valid_index().floor("H")
        #end = df.last_valid_index().ceil("H")
    
        axes.set_xticks(mdates.HourLocator().tick_values(start1, end1))
        axes.set_xticklabels(axes.get_xticks(), rotation = 90)
        axes.xaxis.set_major_locator(mdates.HourLocator())
        #ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=time_delta["interval"]))
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        #axes.get_figure().savefig("Greg_Orr_multistats.png")
    return f
def main():
    raw_data_files = ["Data_from_device1.txt","Data_from_device2.txt","Data_from_device3.txt",
                      "Data_from_device4.txt","Data_from_device5.txt","Data_from_device6.txt"]

    processed_data_file = "static/data.json"
    units_for_data_key = {"bits":"Mbps","pkt":"Mpps"}
    savefile = 'throughput_one_day.png'
    database = 'mySQLitedb.db'

    f, axes = plt.subplots(7, 1, figsize=(32,32))
    for i, raw_data_file in enumerate(raw_data_files):
        #df = raw_data_to_pandas(raw_data_file,processed_data_file)
        #create_SQLite_table(database,df,raw_data_file)
        df = read_from_SQL(database,raw_data_file)
        if i == 0:
            start = df.first_valid_index().floor("H")
            end = df.last_valid_index().ceil("H")
            print("try")
            print(start)
            print(end)
        df = df[df["port"] == "1"]
        #create_SQLite_table(database,df)
        #print(df)
        #print(df.info())
        #df = read_from_SQL('Gregg_Orr.db')
        total_seconds = df. shape[0]
        #start = time.perf_counter()
        #summed_df = summarize_to_2000_points_MAX(df)
        #butter_lowpass_filter(df)
        axes[i].plot(df.index,df['bitstx'].values)
        axes[i].plot(df.index,df['bitsrx'].values)
        axes[i].set_xlabel('time')
        axes[i].set_ylabel('Mbps')
        axes[i].set_title('raw ' + raw_data_file)
        axes[i].grid()
        axes[i].legend(loc='upper left')
        print("plot")
        print(start)
        print(end)
        #axes[i].set_xticks(mdates.MinuteLocator(interval=2).tick_values(df.first_valid_index(), df.last_valid_index()))
        axes[i].set_xticks(mdates.HourLocator(interval = 6).tick_values(start, end))
        axes[i].set_xticklabels(axes[i].get_xticks(), rotation = 45)
        axes[i].xaxis.set_major_locator(mdates.HourLocator(interval = 6))
        #ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=time_delta["interval"]))
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

        """summed_df = summarize_to_2000_points_MEAN(df)
        summed_df["diff"] = summed_df["bitstx"].diff()
        Plot_FFTs(df,summed_df)"""
        #simple_plot_full(summed_df,units_for_data_key,savefile)
        #summed_df = summarize_to_2000_points_MAX(df, my_date_range = ['2023-01-18 13:33:28', '2023-01-18 15:28:45'])
        #simple_plot_full(summed_df,units_for_data_key,savefile)
        #print(summed_df)
        #summed_df = summarize_to_2000_points_MEAN(df)
        #summed_df = summarize_to_2000_points_ROLLING(df)
        #summed_df = summarize_to_fewer_points_MAX(df, freq =60)
        #output = output_to_javascript(summed_df)
        #end = time.perf_counter()
        #print(end-start)
        #X = FFT_hour(summed_df,total_seconds/4000)
        #print(X)
        #IFFT(X)
        #create_histogram_full(df,81,51)
        #create_histogram_full(df,101,71)
        #print(summed_df)
        #simple_plot_full(summed_df,units_for_data_key,savefile)
        #simple_plot_full(df,units_for_data_key,savefile)
    raw_data_file = "Data_from_device_Adfirst"
    #df = raw_data_to_pandas(raw_data_file + ".txt",processed_data_file)
    #df = raw_data_to_pandas(raw_data_file + ".txt",processed_data_file)
    #create_SQLite_table(database,df,raw_data_file)
    df = read_from_SQL(database,raw_data_file)
    raw_data_file = "Data_from_device_Ad"
    df1 = raw_data_to_pandas(raw_data_file + ".txt",processed_data_file)
    df = pd.concat([df,df1])
    plot_adtran(df,axes = axes[6],start1 = start,end1 = end)
    f.tight_layout(pad=4.0)
    f.savefig("mySQLitedb_multistats.png")
if __name__ == "__main__":
    #main()

    processed_data_file = "static/data.json"
    units_for_data_key = {"bits":"Mbps","pkt":"Mpps"}
    savefile = 'throughput_one_day.png'
    raw_data_file = "Data_from_device_Ad_first"
    database = 'mySQLitedbMulti.db'
    df = read_from_SQL(database,raw_data_file)
    raw_data_file = "Data_from_device_Ad"
    df1 = raw_data_to_pandas(raw_data_file + ".txt",processed_data_file)
    #create_SQLite_table(database,df,raw_data_file)
    #df = read_from_SQL(database)
    df = pd.concat([df,df1])
    f = plot_adtran(df, separate = True)
    f.tight_layout(pad=4.0)
    f.savefig("device_Ad.png")