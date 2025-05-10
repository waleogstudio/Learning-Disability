import os 
import re
import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import butter, filtfilt
import scipy.stats as stats

def text(name):
    return name

def scaling_process(data):
    scaler=StandardScaler()
    data=scaler.fit_transform(data)
    return data

def split_data(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test

def mimmax_scale(data):
    scale=MinMaxScaler()
    data=scale.fit_transform(data.reshape(data.shape[0],-1)).reshape(data.shape)
    return data


# eye_root_dir='/home/user14/Downloads/Multi Model Learning Disability/eye-tracking/data'

def eye_preprocess(eye_root_dir):
    def eye_call_files(directory,name):
        data=os.path.join(directory,name)
        all_files=os.listdir(data)
        add_link=[(os.path.join(data,i)) for i in all_files]
        return add_link
    
    
    def eye_select_file(files,number):
        selected_files = [file for file in files if str(number) in file]
        return selected_files
    
    eye_all_files=eye_call_files(eye_root_dir,'data')
    selected_1003 = eye_select_file(eye_all_files, 1003)
    selected_1016 = eye_select_file(eye_all_files, 1016)  
    selected_1019 = eye_select_file(eye_all_files, 1019)
    selected_1033 = eye_select_file(eye_all_files, 1033)  
    selected_1040 = eye_select_file(eye_all_files, 1040)   
    selected_1009 = eye_select_file(eye_all_files, 1009)  
    selected_1021 = eye_select_file(eye_all_files, 1021)  
    selected_1038 = eye_select_file(eye_all_files, 1038)
    selected_1082 = eye_select_file(eye_all_files, 1082)  
    selected_1113 = eye_select_file(eye_all_files, 1113)  
    
    
    
    
    def eye_categorize_files(files):
        t1_files = [f for f in files if "_fixations.csv" in f]  
        t4_files = [f for f in files if "_saccades.csv" in f] 
        t5_files = [f for f in files if "_metrics.csv" in f] 
        return t1_files, t4_files, t5_files
    
    fixations_1003,saccades_1003,metrics_1003 = eye_categorize_files(selected_1003)
    fixations_1016,saccades_1016,metrics_1016 = eye_categorize_files(selected_1016)
    fixations_1019,saccades_1019,metrics_1019 = eye_categorize_files(selected_1019)
    fixations_1033,saccades_1033,metrics_1033 = eye_categorize_files(selected_1033)
    fixations_1040,saccades_1040,metrics_1040 = eye_categorize_files(selected_1040)
    fixations_1009,saccades_1009,metrics_1009 = eye_categorize_files(selected_1009)
    fixations_1021,saccades_1021,metrics_1021 = eye_categorize_files(selected_1021)
    fixations_1038,saccades_1038,metrics_1038 = eye_categorize_files(selected_1038)
    fixations_1082,saccades_1082,metrics_1082 = eye_categorize_files(selected_1082)
    fixations_1113,saccades_1113,metrics_1113 = eye_categorize_files(selected_1113)
    
    def eye_read_fixation_data(datas,columns,sample_size=100):
        data_frames = []       
        for file in datas:
            df = pd.read_csv(file, usecols=columns) 
            df_sampled = df.sample(n=min(sample_size, len(df)), random_state=42)  
            data_frames.append(df_sampled)
            
        combined_data = pd.concat(data_frames, axis=0) 
        return combined_data
    fixation_columns_to_extract = ["start_ms","end_ms","duration_ms", "fix_x", "fix_y"]
    fixations_data_1003=eye_read_fixation_data(fixations_1003,fixation_columns_to_extract,100)
    fixations_data_1016=eye_read_fixation_data(fixations_1016,fixation_columns_to_extract,100)
    fixations_data_1019=eye_read_fixation_data(fixations_1019,fixation_columns_to_extract,100)
    fixations_data_1033=eye_read_fixation_data(fixations_1033,fixation_columns_to_extract,100)
    fixations_data_1040=eye_read_fixation_data(fixations_1040,fixation_columns_to_extract,100)
    fixations_data_1009=eye_read_fixation_data(fixations_1009,fixation_columns_to_extract,100)
    fixations_data_1021=eye_read_fixation_data(fixations_1021,fixation_columns_to_extract,100)
    fixations_data_1038=eye_read_fixation_data(fixations_1038,fixation_columns_to_extract,100)
    fixations_data_1082=eye_read_fixation_data(fixations_1082,fixation_columns_to_extract,100)
    fixations_data_1113=eye_read_fixation_data(fixations_1113,fixation_columns_to_extract,100)
    
    
    def eye_read_saccades_data(datas,columns,sample_size=100):
        data_frames = []       
        for file in datas:
            df = pd.read_csv(file, usecols=columns) 
            df_sampled = df.sample(n=min(sample_size, len(df)), random_state=42) 
            data_frames.append(df_sampled)
            
        combined_data = pd.concat(data_frames, axis=0)  
        return combined_data
    
    
    saccad_columns_to_extract = ["ampl_x","ampl_y","avg_vel_x", "avg_vel_y", "ampl"]  
    saccades_data_1003=eye_read_saccades_data(saccades_1003,saccad_columns_to_extract,100)
    saccades_data_1016=eye_read_saccades_data(saccades_1016,saccad_columns_to_extract,100)
    saccades_data_1019=eye_read_saccades_data(saccades_1019,saccad_columns_to_extract,100)
    saccades_data_1033=eye_read_saccades_data(saccades_1033,saccad_columns_to_extract,100)
    saccades_data_1040=eye_read_saccades_data(saccades_1040,saccad_columns_to_extract,100)
    saccades_data_1009=eye_read_saccades_data(saccades_1009,saccad_columns_to_extract,100)
    saccades_data_1021=eye_read_saccades_data(saccades_1021,saccad_columns_to_extract,100)
    saccades_data_1038=eye_read_saccades_data(saccades_1038,saccad_columns_to_extract,100)
    saccades_data_1082=eye_read_saccades_data(saccades_1082,saccad_columns_to_extract,100)
    saccades_data_1113=eye_read_saccades_data(saccades_1113,saccad_columns_to_extract,100)
    
    
    def eye_read_metrics_data(datas,columns,sample_size=100):
        data_frames = []       
        for file in datas:
            df = pd.read_csv(file, usecols=columns) 
            df_sampled = df.sample(n=min(sample_size, len(df)), random_state=42) 
            data_frames.append(df_sampled)
            
        combined_data = pd.concat(data_frames, axis=0)  
        return combined_data
    
    
    metrics_columns_to_extract = ["mean_fix_dur_trial","mean_sacc_dur_trial","mean_sacc_ampl_trial", "mean_fix_dur_aoi"]  
    metrics_data_1003=eye_read_metrics_data(metrics_1003,metrics_columns_to_extract,100)
    metrics_data_1016=eye_read_metrics_data(metrics_1016,metrics_columns_to_extract,100)
    metrics_data_1019=eye_read_metrics_data(metrics_1019,metrics_columns_to_extract,100)
    metrics_data_1033=eye_read_metrics_data(metrics_1033,metrics_columns_to_extract,100)
    metrics_data_1040=eye_read_metrics_data(metrics_1040,metrics_columns_to_extract,100)
    metrics_data_1009=eye_read_metrics_data(metrics_1009,metrics_columns_to_extract,100)
    metrics_data_1021=eye_read_metrics_data(metrics_1021,metrics_columns_to_extract,100)
    metrics_data_1038=eye_read_metrics_data(metrics_1038,metrics_columns_to_extract,100)
    metrics_data_1082=eye_read_metrics_data(metrics_1082,metrics_columns_to_extract,100)
    metrics_data_1113=eye_read_metrics_data(metrics_1113,metrics_columns_to_extract,100)
    
    final_data_1003=np.concatenate((fixations_data_1003,saccades_data_1003,metrics_data_1003),axis=1)
    final_data_1016=np.concatenate((fixations_data_1016,saccades_data_1016,metrics_data_1016),axis=1)
    final_data_1019=np.concatenate((fixations_data_1019,saccades_data_1019,metrics_data_1019),axis=1)
    final_data_1033=np.concatenate((fixations_data_1033,saccades_data_1033,metrics_data_1033),axis=1)
    final_data_1040=np.concatenate((fixations_data_1040,saccades_data_1040,metrics_data_1040),axis=1)
    final_data_1009=np.concatenate((fixations_data_1009,saccades_data_1009,metrics_data_1009),axis=1)
    final_data_1021=np.concatenate((fixations_data_1021,saccades_data_1021,metrics_data_1021),axis=1)
    final_data_1038=np.concatenate((fixations_data_1038,saccades_data_1038,metrics_data_1038),axis=1)
    final_data_1082=np.concatenate((fixations_data_1082,saccades_data_1082,metrics_data_1082),axis=1)
    final_data_1113=np.concatenate((fixations_data_1113,saccades_data_1113,metrics_data_1113),axis=1)
    
    dyslexia=np.concatenate((final_data_1009,final_data_1021,final_data_1038,final_data_1082,final_data_1113),axis=0)
    non_dyslexia=np.concatenate((final_data_1003,final_data_1016,final_data_1019,final_data_1033,final_data_1040),axis=0)
    
    eye_x=np.concatenate((non_dyslexia,dyslexia),axis=0)
    eye_y=np.concatenate((np.zeros(len(non_dyslexia)),np.ones(len(dyslexia))),axis=0)

    eye_x=scaling_process(eye_x)

    
    return dyslexia,non_dyslexia,eye_x,eye_y

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# hw_root_dir = "/home/user14/Downloads/Multi Model Learning Disability/Hand writing/Dataset Dyslexia_Password WanAsy321 (extract.me)/Gambo"
def hand_write(hw_root_dir):
    def hw_train_test_data(directory, name, count=200):
        train_or_test = os.path.join(directory, name)   
        normal = os.path.join(train_or_test, "Normal")
        normal_images = sorted(os.listdir(normal))[:count]  # Take first 'count' images
        normal_links = [os.path.join(normal, i) for i in normal_images]
        reversal = os.path.join(train_or_test, "Reversal")
        print(reversal)
        reversal_images = sorted(os.listdir(reversal))[:count]  # Take first 'count' images
        reversal_links = [os.path.join(reversal, i) for i in reversal_images]
    
        return normal_links, reversal_links
    
    
    def hw_imag_resize(images):
        lst=[]
        for i in images:
            img=Image.open(i).convert("RGB") 
            resize=img.resize((224,224))
            array_img=np.array(resize) 
            if array_img.shape==(224,224,3):
                lst.append(array_img)
    
            else:
                pass
        return lst
    
    
    train_normal, train_reversal = hw_train_test_data(hw_root_dir, "Train", count=1500)
    
    train_normal_resize=hw_imag_resize(train_normal)
    train_reversal_resize=hw_imag_resize(train_reversal)

    
    train_normal_resize=np.array(train_normal_resize)
    train_reversal_resize=np.array(train_reversal_resize)

    handwrite_x=np.concatenate((train_normal_resize,train_reversal_resize),axis=0)

    handwrite_x=mimmax_scale(handwrite_x)
    return train_normal_resize,train_reversal_resize,handwrite_x



# # #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def eeg_add_participants(directory, name):
    participant1 = os.path.join(directory, name)
    session = os.path.join(participant1, 'ses-01')
    eeg = os.path.join(session, 'eeg')
    list_all_files = os.listdir(eeg)
    lst=[os.path.join(eeg,i) for i in list_all_files]
    event_files = sorted([f for f in lst if re.search(r"_run-\d+_events\.tsv$", f)])
    eeg_files = sorted([f for f in lst if re.search(r"_run-\d+_eeg\.set$", f)])
    
    def extract_run_number(filename):
        match = re.search(r"_run-(\d+)_", filename)
        return int(match.group(1)) if match else float('inf')
    
    event_files.sort(key=extract_run_number)
    eeg_files.sort(key=extract_run_number)

    return event_files, eeg_files


def eeg_extract_time_intervals(df):
    start_times_correct = []
    end_times_correct = []
    start_times_non_correct = []
    end_times_non_correct = []

    for i in range(1, len(df)):
        if df.loc[i, "value"] == "correct":
            start_times_correct.append(df.loc[i - 1, "onset"])
            end_times_correct.append(df.loc[i, "onset"])
        else:
            start_times_non_correct.append(df.loc[i - 1, "onset"])
            end_times_non_correct.append(df.loc[i, "onset"])

    return start_times_correct, end_times_correct, start_times_non_correct, end_times_non_correct


def extract_eeg_intervals_to_array(file_path, start_times, end_times):
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    extracted_data = []  
    for start, end in zip(start_times, end_times):
        raw_segment = raw.copy().crop(tmin=start, tmax=end)
        raw_segment.notch_filter(freqs=50, picks="all", method="spectrum_fit", filter_length="auto", phase="zero")
        extracted_data.append(raw_segment)
    
    if extracted_data:
        final_data = mne.concatenate_raws(extracted_data)
        eeg_array = final_data.get_data().T  
    else:
        eeg_array = None 
    return eeg_array


def eeg_ica_process(data):
    ica = FastICA(n_components=31)
    ica_data = ica.fit_transform(data)
    return ica_data

def eeg_ica_plots_show(data_signals,name):
    plt.figure(figsize=(8, 8))
    plt.plot(data_signals)
    plt.title(name)
    plt.xlabel("Time Samples")
    plt.ylabel("Amplitude")
    plt.show()


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')  # Bandpass filter
    return b, a

# Apply Butterworth filter
def butter_bandpass_filter(data, lowcut, highcut, fs=1000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Apply filter to each component (column) of the 2D array
    y = np.zeros_like(data)
    for i in range(data.shape[1]):  # Loop through each component
        y[:, i] = filtfilt(b, a, data[:, i])  # Apply filter with zero phase distortion
    return y

def before_and_after_plots(before,after,before_name,after_name):
    plt.figure(figsize=(10, 4))
    plt.plot(before[:, 0], label=before_name)
    plt.plot(after[:, 0], label=after_name)
    plt.title("EEG Signal Before and After Filtering")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()



eeg_root_dir='/home/user14/Downloads/Multi Model Learning Disability/eeg_data/ds002680'


def eeg_preprocess(eeg_root_dir):

    eeg_event_files_1, eeg_files_1 = eeg_add_participants(eeg_root_dir, "sub-002")
    
    
    time_data1 = pd.read_csv(eeg_event_files_1[0], sep="\t")
    time_data2 = pd.read_csv(eeg_event_files_1[1], sep="\t")
    time_data3 = pd.read_csv(eeg_event_files_1[2], sep="\t")
    time_data4 = pd.read_csv(eeg_event_files_1[3], sep="\t")
    
    start_times_correct_1, end_times_correct_1, start_times_non_correct_1, end_times_non_correct_1 = eeg_extract_time_intervals(time_data1)
    start_times_correct_2, end_times_correct_2, start_times_non_correct_2, end_times_non_correct_2 = eeg_extract_time_intervals(time_data2)
    start_times_correct_3, end_times_correct_3, start_times_non_correct_3, end_times_non_correct_3 = eeg_extract_time_intervals(time_data3)
    start_times_correct_4, end_times_correct_4, start_times_non_correct_4, end_times_non_correct_4 = eeg_extract_time_intervals(time_data4)
    
    number_of_samples=375
    
    correct_eeg_array_1 = extract_eeg_intervals_to_array(eeg_files_1[0], start_times_correct_1, end_times_correct_1)[:number_of_samples]
    non_correct_eeg_array_1 = extract_eeg_intervals_to_array(eeg_files_1[0], start_times_non_correct_1, end_times_non_correct_1)[:number_of_samples]
    
    
    correct_eeg_array_2 = extract_eeg_intervals_to_array(eeg_files_1[1], start_times_correct_2, end_times_correct_2)[:number_of_samples]
    non_correct_eeg_array_2 = extract_eeg_intervals_to_array(eeg_files_1[1], start_times_non_correct_2, end_times_non_correct_2)[:number_of_samples]
    
    
    correct_eeg_array_3 = extract_eeg_intervals_to_array(eeg_files_1[2], start_times_correct_3, end_times_correct_3)[:number_of_samples]
    non_correct_eeg_array_3 = extract_eeg_intervals_to_array(eeg_files_1[2], start_times_non_correct_3, end_times_non_correct_3)[:number_of_samples]
    
    correct_eeg_array_4 = extract_eeg_intervals_to_array(eeg_files_1[3], start_times_correct_4, end_times_correct_4)[:number_of_samples]
    non_correct_eeg_array_4 = extract_eeg_intervals_to_array(eeg_files_1[3], start_times_non_correct_4, end_times_non_correct_4)[:number_of_samples]
    
    
    total_correct=np.concatenate((correct_eeg_array_1,correct_eeg_array_2,correct_eeg_array_3,correct_eeg_array_4),axis=0)
    total_non_correct=np.concatenate((non_correct_eeg_array_1,non_correct_eeg_array_2,non_correct_eeg_array_3,non_correct_eeg_array_4),axis=0)
    
    total_non_correct_ica=eeg_ica_process(total_non_correct)
    total_correct_ica=eeg_ica_process(total_correct)
    
    sampling_rate = 1000  
    # duration = len(total_correct) / sampling_rate

    total_correct_ica = total_correct_ica - np.mean(total_correct, axis=0)
    total_non_correct_ica = total_non_correct_ica - np.mean(total_non_correct, axis=0)

    lowcut = 0.5 
    highcut = 50.0  
    order = 3 

    total_correct_bwf = butter_bandpass_filter(total_correct_ica, lowcut, highcut, sampling_rate, order)
    total_non_correct_bwf = butter_bandpass_filter(total_non_correct_ica, lowcut, highcut, sampling_rate, order)
    
    eeg_x = np.concatenate((total_non_correct_bwf, total_correct_bwf), axis=0)
    eeg_x=scaling_process(eeg_x)

    
    return total_correct ,total_non_correct,total_correct_ica,total_non_correct_ica,total_correct_bwf,total_non_correct_bwf,eeg_x
    

def save_plot(signal, name, idx):
    plt.figure(figsize=(8, 4))
    plt.plot(signal)
    plt.title(f"{name} - Sample {idx+1}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    filepath = f"static/images/eeg_plot_{idx}.png"
    plt.savefig(filepath)
    plt.close()
    return f"eeg_plot_{idx}.png"





def splitdata(eye_x, eeg_x, handwrite_x, eye_y):
    eye_x_train,eye_x_test,eeg_x_train,eeg_x_test,hw_x_train,hw_x_test,y_train,y_test=train_test_split(eye_x,eeg_x,handwrite_x,eye_y,test_size=0.2,random_state=42)

    return eye_x_train,eye_x_test,eeg_x_train,eeg_x_test,hw_x_train,hw_x_test,y_train,y_test


import tensorflow as tf
from tensorflow.keras import layers,Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MultiHeadAttention, Embedding, GlobalAveragePooling1D, Dropout, Reshape, Input,LSTM,Bidirectional



def transformer_model(x_train, y_train, x_test, y_test,verbose=1):
    input_dim=x_train.shape[1]
    num_head=1
    ff_dim=64# Number of features
    
    input_layer = layers.Input(shape=(input_dim,))
    dense_layer = layers.Dense(128, activation='relu')(input_layer)
    reshape_input = layers.Reshape((1, 128))(dense_layer)
    attention_layer = layers.MultiHeadAttention(num_heads=num_head, key_dim=ff_dim)(reshape_input, reshape_input)
    attention_residual = layers.Add()([reshape_input, attention_layer])
    normalized_attention = layers.LayerNormalization()(attention_residual)
    ff_dense1 = layers.Dense(ff_dim, activation='relu')(normalized_attention)
    ff_dense2 = layers.Dense(128)(ff_dense1)
    ff_residual = layers.Add()([normalized_attention, ff_dense2])
    normalized_ff = layers.LayerNormalization()(ff_residual)
    feature_layer = layers.GlobalAveragePooling1D(name="feature_extraction")(normalized_ff)
    output_layer = layers.Dense(1, activation='sigmoid')(feature_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_test, y_test),verbose=0)
    feature_model = Model(inputs=model.input, outputs=feature_layer)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=verbose)
    train_features = feature_model.predict(x_train) 
    test_features = feature_model.predict(x_test)  
    
    return train_features, test_features,model,accuracy


def build_BiLSTM_model(x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
    input_shape = x_train.shape[1]
    
    # Reshape input for LSTM (batch_size, time_steps, features)
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    inputs = Input(shape=(1, input_shape)) 
    x = Bidirectional(LSTM(64, return_sequences=False))(inputs)  
    x = Dense(16, activation='relu')(x)
    feature_layer = Dense(8, activation='relu', name="feature_layer")(x)  
    outputs = Dense(1, activation='sigmoid')(feature_layer)  

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
              validation_data=(x_test, y_test), verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    # Feature extraction
    feature_extractor = Model(inputs=model.input, outputs=feature_layer)
    train_features = feature_extractor.predict(x_train)
    test_features = feature_extractor.predict(x_test)

    return train_features, test_features, accuracy






def create_and_train_model(x_train, x_test, y_train, y_test,
                           patch_size=16, image_size=224,
                           hidden_dim=256, num_heads=1,
                           num_classes=1, epochs=10, batch_size=32):

    input_layer = Input(shape=(image_size, image_size, 3))
    patches = Conv2D(filters=hidden_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(input_layer)
    patches = Reshape(((image_size // patch_size) ** 2, hidden_dim))(patches)
    patch_indices = tf.range((image_size // patch_size) ** 2)
    positional_encoding = Embedding(input_dim=(image_size // patch_size) ** 2, output_dim=hidden_dim)(patch_indices)
    x = patches + positional_encoding  # Adding positional encoding to patch embeddings
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)(x, x)
    x = attention_output + patches  # Skip connection
    x = Dense(hidden_dim, activation='relu')(x)
    x = Dense(hidden_dim, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    features = Flatten()(x)  # Features before the final Dense layer
    output_layer = Dense(num_classes, activation='sigmoid')(features)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(x_test, y_test), verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    feature_extraction_model = Model(inputs=input_layer, outputs=features)
    train_features = feature_extraction_model.predict(x_train)
    test_features = feature_extraction_model.predict(x_test)

    return train_features, test_features, accuracy


def hirarcial_cross_attention(eye_train_features,eeg_train_features,hw_train_features,eye_test_features,eeg_test_features,hw_test_features,y_train,y_test):
    def compute_qkv(features, d_k=64, d_v=64):
        Q = tf.keras.layers.Dense(d_k, activation='relu', name='query')(features)
        K = tf.keras.layers.Dense(d_k, activation='relu', name='key')(features)
        V = tf.keras.layers.Dense(d_v, activation='relu', name='value')(features)
        return Q, K, V


    def cross_attention(Q, K, V):
        d_k = tf.cast(tf.shape(K)[-1], tf.float32) 
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(d_k)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attention_weights, V)
        
        return output
    
    eye_Q, eye_K, eye_V = compute_qkv(eye_train_features)
    eeg_Q, eeg_K, eeg_V = compute_qkv(eeg_train_features)
    hw_Q, hw_K, hw_V = compute_qkv(hw_train_features)

    t_eye_Q, t_eye_K, t_eye_V = compute_qkv(eye_test_features)
    t_eeg_Q, t_eeg_K, t_eeg_V = compute_qkv(eeg_test_features)
    t_hw_Q, t_hw_K, t_hw_V = compute_qkv(hw_test_features)



    train_low=cross_attention(hw_Q,eye_K,eeg_V)
    test_low=cross_attention(t_hw_Q,t_eye_K,t_eeg_V)


    low_Q,low_K,low_V=compute_qkv(train_low)
    t_low_Q,t_low_K,t_low_V=compute_qkv(test_low)


    train_middle=cross_attention(low_Q,eeg_K,eeg_V)
    test_middle=cross_attention(t_low_Q,t_eeg_K,t_eeg_V)

    midd_Q,midd_K,midd_V=compute_qkv(train_middle)
    t_midd_Q,t_midd_K,t_midd_V=compute_qkv(test_middle)


    train_high=cross_attention(midd_Q,hw_K,hw_V)
    test_high=cross_attention(t_midd_Q,t_hw_K,t_hw_V)


    train_high=scaling_process(train_high)
    test_high=scaling_process(test_high)

    x = np.concatenate([train_high, test_high], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)


    x_train, x_test, y_train, y_test = split_data(x, y)
    return x_train, x_test, y_train, y_test



def final_model_process(x_train, x_test, y_train, y_test):
    def transformer_model(x_train, y_train, x_test, y_test, verbose=1):
        input_dim = x_train.shape[1]
        num_head = 1
        ff_dim = 64

        input_layer = layers.Input(shape=(input_dim,))
        dense_layer = layers.Dense(128, activation='relu')(input_layer)
        reshape_input = layers.Reshape((1, 128))(dense_layer)
        attention_layer = layers.MultiHeadAttention(num_heads=num_head, key_dim=ff_dim)(reshape_input, reshape_input)
        attention_residual = layers.Add()([reshape_input, attention_layer])
        normalized_attention = layers.LayerNormalization()(attention_residual)
        ff_dense1 = layers.Dense(ff_dim, activation='relu')(normalized_attention)
        ff_dense2 = layers.Dense(128)(ff_dense1)
        ff_residual = layers.Add()([normalized_attention, ff_dense2])
        normalized_ff = layers.LayerNormalization()(ff_residual)
        feature_layer = layers.GlobalAveragePooling1D(name="feature_extraction")(normalized_ff)
        output_layer = layers.Dense(1, activation='sigmoid')(feature_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        model.fit(x_train, y_train, epochs=10, batch_size=8, validation_data=(x_test, y_test), verbose=0)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=verbose)

        return accuracy, model

    return transformer_model(x_train, y_train, x_test, y_test)


from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, confusion_matrix

def model_metrics(model, test_data, test_label):
    y_pred = model.predict(test_data)
    y_predicted = [1 if i > 0.5 else 0 for i in y_pred]

    acc = accuracy_score(test_label, y_predicted)
    precision = precision_score(test_label, y_predicted)
    f_score = f1_score(test_label, y_predicted)

    tn, fp, fn, tp = confusion_matrix(test_label, y_predicted).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc = roc_auc_score(test_label, y_pred)

    return acc, precision, f_score, sensitivity, specificity, auc, y_predicted








