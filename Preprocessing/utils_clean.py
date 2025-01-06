import pandas as pd
import re 
import numpy as np 
import pyxdf
import mne
import matplotlib.pyplot as plt
from mne.decoding import Scaler


def show_streams(data):

    # n_streams = 3
    # i = 0

    for i, channel_data in enumerate(data):
        # Check if the channel data is a dictionary and has the 'info' key
        if isinstance(channel_data, dict) and "info" in channel_data:
            print(f"Channel {i}:")
            print(f"Effective Rate: {channel_data['info']['effective_srate']}")
            print(f"Start TimeStamp: ")

            # Access the channel name from the 'info' dictionary
            channel_name = channel_data["info"].get("name", "Unknown Channel")  # Use 'get' to avoid KeyError if 'name' is missing
            print(f"Channel Name: {channel_name}")

            # Access other information from the 'info' dictionary as needed
            # For example, to print the channel type:
            channel_type = channel_data["info"].get("type", "Unknown Type")
            print(f"Channel Type: {channel_type}")

            # Access the time series data
            if "time_series" in channel_data:
                time_series = channel_data["time_series"]

                if isinstance(time_series, list):
                    print(f"Time Series Length: {len(time_series)}")
                else:
                    print(f"Time Series Shape: {time_series.shape}")

                # Perform further analysis on the time series data as needed

            print("-" * 20)  # Separator between channels

def find_stream(name, streams):
    if name == 'eeg':
        name = 'EE225-000000-000867-02-DESKTOP-4OD688D'
    elif name == 'marker':
        name = 'LSLMarkersInletStreamName2'

    else:
        assert False

    for stream in streams:
        if stream['info']['name'][0] == name:
            return stream

    assert False

def get_time_series(stream):
    """
    Get time series data for all channels form a specific stream.
    The exact time stamp for each row is added as an additional column.

    Args:
    -------
        stream : One stream.

    Returns:
    -------
        df (pandas.DataFrame): Time series data for all channels.
    """

    if stream['info']['type'][0] == 'Markers':
        columns = ['marker']
    else:
        assert False

    # get time series from this stream
    time_series = stream['time_series']

    # get time stamps as index
    time_stamps = stream['time_stamps']

    # TODO: ADJUST WITH OFFSET

    df = pd.DataFrame(data=time_series, columns=columns, index=time_stamps)

    return df

def get_time_stamps(stream):
    """
    Get time stamp data for all channels form a specific stream.
    The exact time stamp for each row is added as an additional column.
     """
    if stream['info']['type'][0] == 'Markers':
        columns = ['marker']
    else:
        assert False

    # get time stamps as index
    time_stamps = stream['time_stamps']

    # TODO: ADJUST WITH OFFSET

    df = pd.DataFrame(columns=columns, index=time_stamps)

    return df



def get_duration(marker_stream):
    
    # Filter markers starting with 'StimStart' and 'StimEnd'
    df_marker = get_time_series(marker_stream)

    marker_object_onset = df_marker[df_marker['marker'].str.startswith('StimStart')].reset_index()
    marker_object_onset['numb'] = range(len(marker_object_onset))
    marker_object_onset = marker_object_onset.rename(columns={'index': 'start', 'start_time': 'start_time'})

    marker_object_offset = df_marker[df_marker['marker'].str.startswith('StimEnd')].reset_index()
    marker_object_offset['numb'] = range(len(marker_object_offset))
    marker_object_offset = marker_object_offset.rename(columns={'index': 'end', 'end_time': 'end_time'})

    merged_df = pd.merge(marker_object_onset, marker_object_offset, on ='numb')
    df_durations = merged_df.drop(['marker_x','marker_y'],axis=1)

    return df_durations

"""Setting channel names for the 64 channel headset"""

def set_channel_names(data):
    ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 
                    'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 
                    'P4', 'P8', 'POz', 'O1', 'O2', 'EOG', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 
                    'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2', 
                    'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz']

    # Ensure the number of channel names matches the number of channels in the data
    if len(ch_names) == len(data.info['ch_names']):
        # Create a mapping from old to new channel names
        rename_dict = {old_name: new_name for old_name, new_name in zip(data.info['ch_names'], ch_names)}
        
        # Rename the channels
        data.rename_channels(rename_dict)
        
        # Print to verify the new channel names
        print("New channel names:", data.info['ch_names'])
    else:
        print("Error: The number of channel names does not match the number of channels in the data.")


def plot_channel_correlation(eeg_preprocessed, ch_names, min_threshold=0.2, max_threshold=0.99):
    correlation_matrix = np.corrcoef(eeg_preprocessed)
    correlation_matrix = np.abs(correlation_matrix)
    
    # Set diagonal entries to 0
    np.fill_diagonal(correlation_matrix, 0)

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot correlation matrix before bad removal
    correlation_matrix_before = correlation_matrix.copy()
    correlation_matrix_before[correlation_matrix_before < min_threshold] = -1
    correlation_matrix_before[correlation_matrix_before > max_threshold] = 2
    im_before = axs[0].imshow(correlation_matrix_before, cmap='hot', interpolation='none')
    axs[0].set_title('Correlation Matrix Before Removal')

    # Remove bad channels if all correlations are below the threshold for that channel
    bad_channels_indices = np.where(np.all(correlation_matrix < min_threshold, axis=1))[0]
    eeg_removed = np.delete(eeg_preprocessed, bad_channels_indices, axis=0)
    bad_channels = list(np.array(ch_names)[bad_channels_indices])
    print("bad channels:", bad_channels_indices)
    print("Bad channels:", bad_channels)
    correlation_matrix = np.corrcoef(eeg_removed.reshape(eeg_removed.shape[0], -1))
    correlation_matrix = np.abs(correlation_matrix)
    np.fill_diagonal(correlation_matrix, 0)

    # Plot correlation matrix after bad removal
    correlation_matrix_after = correlation_matrix.copy()
    correlation_matrix_after[correlation_matrix_after < min_threshold] = -1
    correlation_matrix_after[correlation_matrix_after > max_threshold] = 2
    im_after = axs[1].imshow(correlation_matrix_after, cmap='hot', interpolation='none')
    axs[1].set_title('Correlation Matrix After Removal')

    # Add colorbar to both heatmaps
    fig.colorbar(im_before, ax=axs[0], fraction=0.046, pad=0.04, label='Correlation Coefficient')
    fig.colorbar(im_after, ax=axs[1], fraction=0.046, pad=0.04, label='Correlation Coefficient')
    
    plt.show()

    return eeg_removed, bad_channels



def detect_bad_channels(eeg_data_filt, ch_names, flat_thresh: float = 1e-9, min_corr_threshold: float = 0.2, max_corr_threshold: float = 0.99):
    """
    Detects bad channels based on NaN values, flat channels, and correlation patterns.
    
    Parameters:
    -----------
    eeg_data_filt : np.ndarray
        3D array of EEG data (channels × time points).
    ch_names : list
        List of channel names corresponding to the data.
    flat_thresh : float, optional
        Threshold for detecting flat channels based on MAD and standard deviation.
    min_corr_threshold : float, optional
        Minimum correlation threshold for detecting poorly correlated channels.
    max_corr_threshold : float, optional
        Maximum correlation threshold for detecting highly correlated channels.

    Returns:
    --------
    bad_channel_names : list
        List of bad channel names detected by NaN, flat signal, and correlation checks.
    """

    # 1. Detect bad channels by NaN values
    def bad_by_nan(eeg_data_filt):
        bad_channels = []
        for i in range(eeg_data_filt.shape[0]):  # Iterate over channels
            if np.isnan(eeg_data_filt[i, :]).any():  # Check for NaN in any time point for this channel
                bad_channels.append(i)
        return bad_channels

    # 2. Detect bad channels by flat signal
    def bad_by_flat(eeg_data_filt, flat_thresh):
        mad = np.median(np.abs(eeg_data_filt - np.median(eeg_data_filt, axis=1, keepdims=True)), axis=1) < flat_thresh
        std = np.std(eeg_data_filt, axis=1) < flat_thresh
        flat_channels = np.where(np.logical_or(mad, std))[0]
        return flat_channels

    # 3. Detect bad channels by correlation
    def check_channel_correlation(eeg_data_filt, min_threshold, max_threshold):
        correlation_matrix = np.corrcoef(eeg_data_filt)
        correlation_matrix = np.abs(correlation_matrix)
        np.fill_diagonal(correlation_matrix, 0)
        
        bad_channels_indices = np.where(np.all(correlation_matrix < min_threshold, axis=1) | np.all(correlation_matrix > max_threshold, axis=1))[0]
        return bad_channels_indices

    # Get bad channels from each method
    nan_bad_channels = bad_by_nan(eeg_data_filt)
    flat_bad_channels = bad_by_flat(eeg_data_filt, flat_thresh)
    corr_bad_channels = check_channel_correlation(eeg_data_filt, min_corr_threshold, max_corr_threshold)

    # Combine all bad channels and remove duplicates
    all_bad_channels = list(set(nan_bad_channels) | set(flat_bad_channels) | set(corr_bad_channels))

    # Convert indices to channel names
    bad_channel_names = [ch_names[i] for i in all_bad_channels]

    return bad_channel_names


def plot_bads(raw_removed, bad_channel_names,):
    bad_channels_regex = '|'.join(bad_channel_names)
    raw_removed.set_montage('standard_1020', on_missing='warn')
    picks = mne.pick_channels_regexp(raw_removed.ch_names, regexp=bad_channels_regex)
    raw_removed.plot(order=picks, n_channels=len(picks), scalings={'eeg': 1e-4})

def ica_analysis(raw_removed):
    ica = mne.preprocessing.ICA(n_components=10, random_state=42, max_iter="auto")
    ica.fit(raw_removed)
    ica.exclude = [0,1,2,3,4,5,6]  # details on how we picked these are omitted here
    raw_reconstructed = raw_removed.copy()
    ica.apply(raw_reconstructed)
    return raw_reconstructed

def ica_plot(raw_reconstructed):
    # ica = mne.preprocessing.ICA(n_components=10, random_state=42, max_iter="auto")
    # ica.fit(raw_removed)
    # ica.exclude = [0,1,2,3,4,5,6,7,8,9]  # details on how we picked these are omitted here
    # raw_reconstructed = raw_removed.copy()
    ica.apply(raw_reconstructed)
    ica.plot_sources(raw_removed, show_scrollbars=False)
    ica.plot_components()
    ica.plot_properties(raw_removed, picks = [0,1,2,3,4,5,6,7,8,9]) 
    ica.plot_overlay(raw_removed, exclude=[0], picks="eeg")
    raw_removed.plot_sensors(show_names=True)

    explained_var_ratio = ica.get_explained_variance_ratio(raw_removed)
    for channel_type, ratio in explained_var_ratio.items():
        print(
            f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
        )
    explained_var_ratio = ica.get_explained_variance_ratio(raw_removed, components=[0], ch_type="eeg")

    # This time, print as percentage
    ratio_percent = round(100 * explained_var_ratio["eeg"])
    print(
        f"Fraction of variance in EEG signal explained by first component: "
        f"{ratio_percent}%")
    
def create_dataset(raw_reconstructed,eeg_stream,marker_stream,df_marker):

    #Sometimes Lab Recorder accidentally adds another empty stream to the xdf file, so we need to check and get rid of it
    # we had 3 incoming channels: EE225-000000-000867-02-DESKTOP-, EE225-000000-000867-02-TRG-DESK, LSLMarkersInletStreamName2 (MSI)
    # if len(data) > 3:
    #     for i, _ in enumerate(data):
    #         if 0 in data[i]["time_series"].shape:
    #             data.pop(i)
        
    # eeg_stream = find_stream('eeg', data)
    # eeg_time_series = eeg_stream["time_series"] #EEG time series data.
    eeg_time_series = raw_reconstructed.get_data()
    eeg_time_series = eeg_time_series.T

    # eeg_data = eeg_stream["time_series"].T
    eeg_timestamps = eeg_stream["time_stamps"] #Timestamps for each sample in the EEG time series.
    # timeseries and stamps for all markers
    event_time_series= marker_stream['time_series']
    event_time_stamps= marker_stream['time_stamps']


    event_time_series_onset = df_marker[df_marker['marker'].str.startswith('StimStart')].reset_index()
    event_time_series_onset['numb'] = range(len(event_time_series_onset))
    event_time_series_onset = event_time_series_onset.rename(columns={'time_stamps': 'start', 'start_time': 'start_time'})

    event_time_series_offset = df_marker[df_marker['marker'].str.startswith('StimEnd')].reset_index()
    event_time_series_offset['numb'] = range(len(event_time_series_offset))
    event_time_series_offset = event_time_series_offset.rename(columns={'time_stamps': 'end', 'end_time': 'end_time'})



    diff = np.subtract.outer(event_time_stamps, eeg_timestamps)
    eeg_indices = np.argmin(np.abs(diff), axis=1)
    eeg_indices_onset= eeg_indices[::2] #every second index because we have markers for StimStart and StimEnd consecutivley
    start = eeg_indices_onset[0] #index at which first event aligns within the EEG 
    # Initialize an array to hold the labels for the EEG samples
    labels = np.empty(eeg_timestamps.shape, dtype=int)
    labels.fill(900) #Fill with 900 (break)
    group_labels = np.copy(labels)
    trials = np.copy(labels)

    # Grab the EEG data from first event onwards and turn into dataframe
    eeg_data = eeg_time_series[start:, :64]

    # montage = mne.channels.make_standard_montage('biosemi64')
    # ch_names_set = montage.ch_names

    ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 
                    'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 
                    'P4', 'P8', 'POz', 'O1', 'O2', 'EOG', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 
                    'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2', 
                    'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz']
    out = pd.DataFrame(eeg_data, columns=ch_names )


    # Mapping for group labels
    group_mapping = {
        'banana': 1,
        'strawberry': 2,
        'panda': 3,
        'basketball': 4,
        'face': 5,
        'tiger': 6
        }

    # Function to map group and specific labels to integers
    def map_group_and_specific_labels(marker):
        # Split the marker string and extract the relevant parts
        parts = marker.split()
        group_label_str = ''.join(filter(str.isalpha, parts[1]))  # Extract only the alphabetic part
        specific_label_str = parts[1]
        
        # Map group label
        group_label = group_mapping.get(group_label_str, 0)
        
        # Map specific label by adhering group label and numeric part as a string
        if group_label > 0:
            specific_number_str = ''.join(filter(str.isdigit, specific_label_str))
            specific_label = int(f"{group_label}{specific_number_str}")  # e.g., banana1 -> 11, banana12 -> 112
        else:
            specific_label = 0
        
        return group_label, specific_label

    # Apply the mapping function to each row
    df_marker[['group_label_int', 'specific_label_int']] = df_marker['marker'].apply(
        lambda x: pd.Series(map_group_and_specific_labels(x))
    )

    # Extract the group and specific labels as strings for clarity
    df_marker['group_label'] = df_marker['marker'].apply(lambda x: ''.join(filter(str.isalpha, x.split()[1])))
    df_marker['specific_label'] = df_marker['marker'].apply(lambda x: x.split()[1])



    df_marker_onset = df_marker[::2]

    # Initialize a dictionary to keep track of the trial counts for each specific label
    trial_counters = {}

    # Function to assign trial labels with incremental counts for each occurrence of the same specific label
    def assign_trial_label(row):
        specific_label = row['specific_label_int']
        
        # Increment the trial number every time we see the specific label
        if specific_label in trial_counters:
            trial_counters[specific_label] += 1
        else:
            trial_counters[specific_label] = 1
        
        # Return the trial label with the current trial count
        return int(f"{specific_label}{trial_counters[specific_label]}")

    # Apply the trial label assignment to each row
    df_marker_onset['trial_label'] = df_marker_onset.apply(assign_trial_label, axis=1)

    # Create the final DataFrame with all relevant columns
    df_int_label_onset = df_marker_onset[['marker', 'group_label', 'specific_label', 'group_label_int', 'specific_label_int', 'trial_label']]


    # Iterate over the events
    event_time=8
    fs=512
    # df_int_label_onset = df_int_label[::2]


    for i in range(len(eeg_indices_onset)-1):
        # Label all EEG samples + 2s after the event with the event's label, rest is Break (900)
        event_end = int(eeg_indices_onset[i] + event_time*fs)
        group_labels[eeg_indices_onset[i]:event_end] = int(df_int_label_onset.iloc[i]['group_label_int'])
        labels[eeg_indices_onset[i]:event_end] = int(df_int_label_onset.iloc[i]['specific_label_int'])
        trials[eeg_indices_onset[i]:event_end] = int (df_int_label_onset.iloc[i]['trial_label'])
        #double check with haojun for the first three lines 

    dataset=[]

    df = pd.concat([pd.DataFrame({'time': eeg_timestamps[start:], 'label': labels[start:], 'group_label': group_labels[start:], 'trial_label':trials[start:]}), out], axis=1)
    df = df[:int(eeg_indices_onset[-1]+event_time*fs)] #Remove the rest of the EEG samples that don't have a label

    dataset.append(df)

    return dataset


def remove_breaks(dataset):
    #remove the breaks
    for i in range(len(dataset)):
        dataset[i] = dataset[i][dataset[i]['group_label'] < 900]
    return dataset 

def return_dataset(dataset,ch_names):
    # #Double check with haojun
    out = []
    for df in dataset:
        grouped = df.groupby('trial_label')
        trial_label = np.array([name for name, group in grouped])
        labels = np.array([group['label'].iloc[0] for name, group in grouped])
        group_labels = np.array([group['group_label'].iloc[0] for name, group in grouped])
        eeg_matrix = np.array([group[ch_names].values for name, group in grouped])        

    out.append((trial_label, labels, group_labels, eeg_matrix))
    return out 

def build_class_epochs_mne(out,sfreq,ch_names,bad_channels): 
    ## Build epochs for the 6 classes 
    out_t = np.swapaxes(out[0][3], 1, 2)

    events = np.column_stack((np.arange(467), np.zeros(467, dtype=int), np.array(out[0][2])))

    event_dict ={'banana': 1,
        'strawberry': 2,
        'panda': 3,
        'basketball': 4,
        'face': 5,
        'tiger': 6 }
    eeg_info = mne.create_info(ch_names=ch_names,sfreq=sfreq,ch_types=["eeg"]*64)
    eeg_epochs_classes= mne.EpochsArray(out_t,eeg_info,events=events,event_id=event_dict)
    eeg_epochs_classes.info['bads']= bad_channels
    return eeg_epochs_classes

def build_cat_epochs_mne(out,sfreq,ch_names,bad_channels):
        ## Built Epochs for each category in the class 
    out_t = np.swapaxes(out[0][3], 1, 2)
    events_cat = np.column_stack((np.arange(468), np.zeros(468, dtype=int), np.array(out[0][1])))
    event_cat_id = {
        'banana1': 11, 'banana2': 12, 'banana3': 13, 'banana4': 14, 'banana5': 15, 
        'banana6': 16, 'banana7': 17, 'banana8': 18, 'banana9': 19, 'banana10': 110, 
        'banana11': 111, 'banana12': 112, 'banana13': 113,
        'strawberry1': 21, 'strawberry2': 22, 'strawberry3': 23, 'strawberry4': 24, 
        'strawberry5': 25, 'strawberry6': 26, 'strawberry7': 27, 'strawberry8': 28, 
        'strawberry9': 29, 'strawberry10': 210, 'strawberry11': 211, 'strawberry12': 212, 
        'strawberry13': 213,
        'panda1': 31, 'panda2': 32, 'panda3': 33, 'panda4': 34, 'panda5': 35, 
        'panda6': 36, 'panda7': 37, 'panda8': 38 , 'panda9': 39, 'panda10': 310, 
        'panda11': 311, 'panda12': 312, 'panda13': 313,
        'basketball1': 41, 'basketball2': 42, 'basketball3': 43, 'basketball4': 44, 
        'basketball5': 45, 'basketball6': 46, 'basketball7': 47, 'basketball8': 48, 
        'basketball9': 49, 'basketball10': 410, 'basketball11': 411, 'basketball12': 412, 
        'basketball13': 413,
        'face1': 51, 'face2': 52, 'face3': 53, 'face4': 54, 'face5': 55, 
        'face6': 56, 'face7': 57, 'face8': 58, 'face9': 59, 'face10': 510, 
        'face11': 511, 'face12': 512, 'face13': 513,
        'tiger1': 61, 'tiger2': 62, 'tiger3': 63, 'tiger4': 64, 'tiger5': 65, 
        'tiger6': 66, 'tiger7': 67, 'tiger8': 68, 'tiger9': 69, 'tiger10': 610, 
        'tiger11': 611, 'tiger12': 612, 'tiger13': 613
    }
    eeg_info = mne.create_info(ch_names=ch_names,sfreq=sfreq,ch_types=["eeg"]*64)
    eeg_epochs_objects= mne.EpochsArray(out_t,eeg_info,events=events_cat,event_id=event_cat_id)
    eeg_epochs_objects.info['bads']= bad_channels
    return eeg_epochs_objects

def plot_topo(eeg_epochs,eeg_info):
    # epochs = mne.EpochsArray(eeg_epochs, eeg_info)
    eeg_epochs.set_montage('standard_1020', on_missing='warn')
    eeg_epochs.compute_psd().plot_topomap(ch_type='eeg', normalize=True)


## EEG data across time for each channel, where each channel's data is averaged across all epochs.
def plot_eeg(eeg_epochs,eeg_info):
    # epochs = mne.EpochsArray(eeg_epochs, eeg_info)
    eeg_epochs.set_montage('standard_1020', on_missing='warn')
    eeg_epochs.plot_image(combine='mean', picks='eeg')


def plot_evoked(eeg_epochs, eeg_info, event_name):
    """
    Plot the evoked potential for a specific class.

    Parameters:
    -----------
    eeg_epochs : mne.EpochsArray
        The epochs array containing EEG data.
    eeg_info : mne.Info
        The info object containing metadata about the EEG recording.
    event_name : str
        The name of the event/class to plot the evoked potential for.
    """
    eeg_epochs.set_montage('standard_1020', on_missing='warn')
    if event_name in eeg_epochs.event_id:
        # Plot the average evoked potential for the specified class
        evoked = eeg_epochs[event_name].average().crop(0,1)
        evoked.plot(picks="eeg", spatial_colors=True)
    else:
        print(f"Event '{event_name}' not found in event_id. Please check the event name.")

def standartization(out):
    out_t = np.swapaxes(out[0][3], 1, 2)
    scaler = Scaler(scalings='mean')
    eeg_epochs_standardized = scaler.fit_transform(out_t)
    eeg_epochs_standardized.shape
    return eeg_epochs_standardized




import numpy as np
import pandas as pd

def labelS02(event_time, subset_Stim_onset, eeg_timestamps, eeg_time_series):
    # EEG channel names
    ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 
                'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 
                'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'EOG', 'AF7', 'AF3', 'AF4', 'AF8', 
                'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 
                'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 
                'TP7', 'TP8', 'PO7', 'PO8', 'Oz']
    
    sub_event_time_stamps = np.array(subset_Stim_onset['index'])

    # Compute the difference between event timestamps and EEG timestamps
    diff = np.subtract.outer(sub_event_time_stamps, eeg_timestamps)
    eeg_indices_onset = np.argmin(np.abs(diff), axis=1)
    
    # Get EEG data within the time window of interest (start to end)
    start = eeg_indices_onset[0]
    end = eeg_indices_onset[-1]
    eeg_sub = eeg_time_series[start:end, :64]  # Assuming 64 channels

    # Trial counters (for labeling)
    trial_counters = {}

    def assign_trial_label(row):
        specific_label = row['specific_label_int']
        if specific_label in trial_counters:
            trial_counters[specific_label] += 1
        else:
            trial_counters[specific_label] = 1
        
        return int(f"{specific_label}{trial_counters[specific_label]}")

    # Apply trial labeling to Stim_onset DataFrame
    subset_Stim_onset['trial_label'] = subset_Stim_onset.apply(assign_trial_label, axis=1)
    sub_df = subset_Stim_onset[['marker', 'group_label_int', 'specific_label_int', 'trial_label']]

    # Create empty arrays for labels, initialized with a default value
    labels = np.full(eeg_timestamps.shape, 900)  # 900 for "Break"
    group_labels = np.copy(labels)
    trials = np.copy(labels)

    fs = 512  # Sampling rate in Hz
    event_time = 8
    # for i in range(len(eeg_indices_onset)):
    #     event_end = int(eeg_indices_onset[i] + event_time * fs)
    #     group_labels[eeg_indices_onset[i]:event_end] = int(sub_df.iloc[i]['group_label_int'])
    #     labels[eeg_indices_onset[i]:event_end] = int(sub_df.iloc[i]['specific_label_int'])
    #     trials[eeg_indices_onset[i]:event_end] = int(sub_df.iloc[i]['trial_label'])

    # Combine EEG data with labels and timestamps into a DataFrame
    # df = pd.concat([pd.DataFrame({
    #     'time': eeg_timestamps[eeg_indices_onset[0]:], 
    #     'label': labels[eeg_indices_onset[0]:], 
    #     'group_label': group_labels[eeg_indices_onset[0]:], 
    #     'trial_label': trials[eeg_indices_onset[0]:]
    # }), pd.DataFrame(eeg_sub, columns=ch_names)], axis=1)
    # Combine EEG data with labels and timestamps into a DataFrame
    # df = pd.concat([pd.DataFrame({
    #     'time': eeg_timestamps[eeg_indices_onset[0]:], 
    #     'label': labels[eeg_indices_onset[0]:], 
    #     'group_label': group_labels[eeg_indices_onset[0]:], 
    #     'trial_label': np.repeat(trials[eeg_indices_onset[0]], len(eeg_timestamps[eeg_indices_onset[0]:]))
    # }), pd.DataFrame(eeg_sub, columns=ch_names)], axis=1)


    # # Trim the DataFrame to remove extra rows
    # df = df[:int(eeg_indices_onset[-1] + event_time * fs)]
    # df = df.iloc[:len(eeg_sub)]  # Ensure proper length based on EEG data
    

    # Iterate over the events
    event_time=8
    fs=512
    # df_int_label_onset = df_int_label[::2]


    for i in range(len(eeg_indices_onset)):
        # Label all EEG samples + 2s after the event with the event's label, rest is Break (900)
        event_end = int(eeg_indices_onset[i] + event_time*fs)
        group_labels[eeg_indices_onset[i]:event_end] = int(sub_df.iloc[i]['group_label_int'])
        labels[eeg_indices_onset[i]:event_end] = int(sub_df.iloc[i]['specific_label_int'])
        trials[eeg_indices_onset[i]:event_end] = int (sub_df.iloc[i]['trial_label'])
        #double check with haojun for the first three lines 

    dataset=[]
    out = pd.DataFrame(eeg_sub, columns=ch_names )

    df = pd.concat([pd.DataFrame({'time': eeg_timestamps[start:], 'label': labels[start:], 'group_label': group_labels[start:], 'trial_label':trials[start:]}), out], axis=1)
    df = df[:int(eeg_indices_onset[-1]+event_time*fs)] #Remove the rest of the EEG samples that don't have a label

    dataset.append(df)
    # dataset = [df]  # Store the DataFrame as part of a dataset list
    return dataset

# def labelS02(event_time, subset_Stim_onset,eeg_timestamps, eeg_time_series):

#     ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 
#                     'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 
#                     'P4', 'P8', 'POz', 'O1', 'O2', 'EOG', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 
#                     'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2', 
#                     'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz']
#     sub_event_time_stamps = subset_Stim_onset['index']
#     sub_event_time_stamps = np.array(sub_event_time_stamps)

#     diff = np.subtract.outer(sub_event_time_stamps, eeg_timestamps)
#     eeg_indices_onset = np.argmin(np.abs(diff), axis=1)
#     start= eeg_indices_onset[0]
#     end = eeg_indices_onset[-1]

#     eeg_sub = eeg_time_series[start:end, :64]
    

#     def assign_trial_label(row):
#         trial_counters = {}
#         specific_label = row['specific_label_int']
        
#         # Increment the trial number every time we see the specific label
#         if specific_label in trial_counters:
#             trial_counters[specific_label] += 1
#         else:
#             trial_counters[specific_label] = 1
        
#         # Return the trial label with the current trial count
#         return int(f"{specific_label}{trial_counters[specific_label]}")
#     # Apply the function to extract group and specific labels

#     out = pd.DataFrame(eeg_sub, columns=ch_names )
#     subset_Stim_onset['trial_label'] = subset_Stim_onset.apply(assign_trial_label, axis=1)
#     sub_df= subset_Stim_onset[['marker','group_label_int', 'specific_label_int','trial_label']]

#     """
#         - event_time: has to be set to the time of the specific event
#         - sub_marker : this is the subset of the marker dataframe for a specific event 
#         - out: stores the eeg_timeseries for the specific part and event and ch_names
#         - eeg_indices_onset: contain all the eeg_timestamps for onset of events
#         - eeg_timestamps: timestamps of eeg data
    
#     """
    
#     labels = np.empty(eeg_timestamps.shape, dtype=int)
#     labels.fill(900) #Fill with 900 (break)
#     group_labels = np.copy(labels)
#     trials = np.copy(labels)
#     fs=512
#     # df_int_label_onset = df_int_label[::2]
#     for i in range(len(eeg_indices_onset)):
#         # Label all EEG samples + 2s after the event with the event's label, rest is Break (900)
#         event_end = int(eeg_indices_onset[i] + event_time*fs)
#         group_labels[eeg_indices_onset[i]:event_end] = int(sub_df.iloc[i]['group_label_int'])
#         labels[eeg_indices_onset[i]:event_end] = int(sub_df.iloc[i]['specific_label_int'])
#         trials[eeg_indices_onset[i]:event_end] = int (sub_df.iloc[i]['trial_label'])
#         #double check with haojun for the first three lines 

#     dataset=[]

#     df = pd.concat([pd.DataFrame({'time': eeg_timestamps[eeg_indices_onset[0]:], 'label': labels[eeg_indices_onset[0]:], 'group_label': group_labels[eeg_indices_onset[0]:], 'trial_label':trials[eeg_indices_onset[0]:]}), out], axis=1)
#     df = df[:int(eeg_indices_onset[-1]+event_time*fs)] #Remove the rest of the EEG samples that don't have a label
#     df = df.iloc[:len(out)]
#     dataset.append(df)  

#     return dataset




# import pandas as pd
# import re
# import numpy as np
# import pyxdf
# import mne
# import matplotlib.pyplot as plt
# from mne.decoding import Scaler


# def show_streams(data):
#     # n_streams = 3
#     # i = 0

#     for i, channel_data in enumerate(data):
#         # Check if the channel data is a dictionary and has the 'info' key
#         if isinstance(channel_data, dict) and "info" in channel_data:
#             print(f"Channel {i}:")
#             print(f"Effective Rate: {channel_data['info']['effective_srate']}")
#             print(f"Start TimeStamp: ")

#             # Access the channel name from the 'info' dictionary
#             channel_name = channel_data["info"].get("name", "Unknown Channel")  # Use 'get' to avoid KeyError if 'name' is missing
#             print(f"Channel Name: {channel_name}")

#             # Access other information from the 'info' dictionary as needed
#             # For example, to print the channel type:
#             channel_type = channel_data["info"].get("type", "Unknown Type")
#             print(f"Channel Type: {channel_type}")

#             # Access the time series data
#             if "time_series" in channel_data:
#                 time_series = channel_data["time_series"]

#                 if isinstance(time_series, list):
#                     print(f"Time Series Length: {len(time_series)}")
#                 else:
#                     print(f"Time Series Shape: {time_series.shape}")

#                 # Perform further analysis on the time series data as needed

#             print("-" * 20)  # Separator between channels


# def find_stream(name, streams):
#     if name == 'eeg':
#         name = 'EE225-000000-000867-02-DESKTOP-4OD688D'
#     elif name == 'marker':
#         name = 'LSLMarkersInletStreamName2'

#     else:
#         assert False

#     for stream in streams:
#         if stream['info']['name'][0] == name:
#             return stream

#     assert False


# def get_time_series(stream):
#     """
#     Get time series data for all channels form a specific stream.
#     The exact time stamp for each row is added as an additional column.

#     Args:
#     -------
#         stream : One stream.

#     Returns:
#     -------
#         df (pandas.DataFrame): Time series data for all channels.
#     """

#     if stream['info']['type'][0] == 'Markers':
#         columns = ['marker']
#     else:
#         assert False

#     # get time series from this stream
#     time_series = stream['time_series']

#     # get time stamps as index
#     time_stamps = stream['time_stamps']

#     # TODO: ADJUST WITH OFFSET

#     df = pd.DataFrame(data=time_series, columns=columns, index=time_stamps)

#     return df


# def get_time_stamps(stream):
#     """
#     Get time stamp data for all channels form a specific stream.
#     The exact time stamp for each row is added as an additional column.
#      """
#     if stream['info']['type'][0] == 'Markers':
#         columns = ['marker']
#     else:
#         assert False

#     # get time stamps as index
#     time_stamps = stream['time_stamps']

#     # TODO: ADJUST WITH OFFSET

#     df = pd.DataFrame(columns=columns, index=time_stamps)

#     return df


# def get_duration(marker_stream):
#     # Filter markers starting with 'StimStart' and 'StimEnd'
#     df_marker = get_time_series(marker_stream)

#     marker_object_onset = df_marker[df_marker['marker'].str.startswith('StimStart')].reset_index()
#     marker_object_onset['numb'] = range(len(marker_object_onset))
#     marker_object_onset = marker_object_onset.rename(columns={'index': 'start', 'start_time': 'start_time'})

#     marker_object_offset = df_marker[df_marker['marker'].str.startswith('StimEnd')].reset_index()
#     marker_object_offset['numb'] = range(len(marker_object_offset))
#     marker_object_offset = marker_object_offset.rename(columns={'index': 'end', 'end_time': 'end_time'})

#     merged_df = pd.merge(marker_object_onset, marker_object_offset, on='numb')
#     df_durations = merged_df.drop(['marker_x', 'marker_y'], axis=1)

#     return df_durations


# """Setting channel names for the 64 channel headset"""


# def set_channel_names(data):
#     ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1',
#                 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz',
#                 'P4', 'P8', 'POz', 'O1', 'O2', 'EOG', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2',
#                 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2',
#                 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz']

#     # Ensure the number of channel names matches the number of channels in the data
#     if len(ch_names) == len(data.info['ch_names']):
#         # Create a mapping from old to new channel names
#         rename_dict = {old_name: new_name for old_name, new_name in zip(data.info['ch_names'], ch_names)}

#         # Rename the channels
#         data.rename_channels(rename_dict)

#         # Print to verify the new channel names
#         print("New channel names:", data.info['ch_names'])
#     else:
#         print("Error: The number of channel names does not match the number of channels in the data.")


# def plot_channel_correlation(eeg_preprocessed, ch_names, min_threshold=0.2, max_threshold=0.99):
#     correlation_matrix = np.corrcoef(eeg_preprocessed)
#     correlation_matrix = np.abs(correlation_matrix)

#     # Set diagonal entries to 0
#     np.fill_diagonal(correlation_matrix, 0)

#     fig, axs = plt.subplots(1, 2, figsize=(15, 7))

#     # Plot correlation matrix before bad removal
#     correlation_matrix_before = correlation_matrix.copy()
#     correlation_matrix_before[correlation_matrix_before < min_threshold] = -1
#     correlation_matrix_before[correlation_matrix_before > max_threshold] = 2
#     im_before = axs[0].imshow(correlation_matrix_before, cmap='hot', interpolation='none')
#     axs[0].set_title('Correlation Matrix Before Removal')

#     # Remove bad channels if all correlations are below the threshold for that channel
#     bad_channels_indices = np.where(np.all(correlation_matrix < min_threshold, axis=1))[0]
#     eeg_removed = np.delete(eeg_preprocessed, bad_channels_indices, axis=0)
#     bad_channels = list(np.array(ch_names)[bad_channels_indices])
#     print("bad channels:", bad_channels_indices)
#     print("Bad channels:", bad_channels)
#     correlation_matrix = np.corrcoef(eeg_removed.reshape(eeg_removed.shape[0], -1))
#     correlation_matrix = np.abs(correlation_matrix)
#     np.fill_diagonal(correlation_matrix, 0)

#     # Plot correlation matrix after bad removal
#     correlation_matrix_after = correlation_matrix.copy()
#     correlation_matrix_after[correlation_matrix_after < min_threshold] = -1
#     correlation_matrix_after[correlation_matrix_after > max_threshold] = 2
#     im_after = axs[1].imshow(correlation_matrix_after, cmap='hot', interpolation='none')
#     axs[1].set_title('Correlation Matrix After Removal')

#     # Add colorbar to both heatmaps
#     fig.colorbar(im_before, ax=axs[0], fraction=0.046, pad=0.04, label='Correlation Coefficient')
#     fig.colorbar(im_after, ax=axs[1], fraction=0.046, pad=0.04, label='Correlation Coefficient')

#     plt.show()

#     return eeg_removed, bad_channels


# def detect_bad_channels(eeg_data_filt, ch_names, flat_thresh: float = 1e-9, min_corr_threshold: float = 0.2, max_corr_threshold: float = 0.99):
#     """
#     Detects bad channels based on NaN values, flat channels, and correlation patterns.
    
#     Parameters:
#     -----------
#     eeg_data_filt : np.ndarray
#         3D array of EEG data (channels × time points).
#     ch_names : list
#         List of channel names corresponding to the data.
#     flat_thresh : float, optional
#         Threshold for detecting flat channels based on MAD and standard deviation.
#     min_corr_threshold : float, optional
#         Minimum correlation threshold for detecting poorly correlated channels.
#     max_corr_threshold : float, optional
#         Maximum correlation threshold for detecting highly correlated channels.

#     Returns:
#     --------
#     bad_channel_names : list
#         List of bad channel names detected by NaN, flat signal, and correlation checks.
#     """

#     # 1. Detect bad channels by NaN values
#     def bad_by_nan(eeg_data_filt):
#         bad_channels = []
#         for i in range(eeg_data_filt.shape[0]):  # Iterate over channels
#             if np.isnan(eeg_data_filt[i, :]).any():  # Check for NaN in any time point for this channel
#                 bad_channels.append(i)
#         return bad_channels

#     # 2. Detect bad channels by flat signal
#     def bad_by_flat(eeg_data_filt, flat_thresh):
#         mad = np.median(np.abs(eeg_data_filt - np.median(eeg_data_filt, axis=1, keepdims=True)), axis=1) < flat_thresh
#         std = np.std(eeg_data_filt, axis=1) < flat_thresh
#         flat_channels = np.where(np.logical_or(mad, std))[0]
#         return flat_channels

#     # 3. Detect bad channels by correlation
#     def check_channel_correlation(eeg_data_filt, min_threshold, max_threshold):
#         correlation_matrix = np.corrcoef(eeg_data_filt)
#         correlation_matrix = np.abs(correlation_matrix)
#         np.fill_diagonal(correlation_matrix, 0)

#         bad_channels_indices = np.where(np.all(correlation_matrix < min_threshold, axis=1) | np.all(correlation_matrix > max_threshold, axis=1))[0]
#         return bad_channels_indices

#     # Get bad channels from each method
#     nan_bad_channels = bad_by_nan(eeg_data_filt)
#     flat_bad_channels = bad_by_flat(eeg_data_filt, flat_thresh)
#     corr_bad_channels = check_channel_correlation(eeg_data_filt, min_corr_threshold, max_corr_threshold)

#     # Combine all bad channels and remove duplicates
#     all_bad_channels = list(set(nan_bad_channels) | set(flat_bad_channels) | set(corr_bad_channels))

#     # Convert indices to channel names
#     bad_channel_names = [ch_names[i] for i in all_bad_channels]

#     return bad_channel_names


# def plot_bads(raw_removed, bad_channel_names, ):
#     bad_channels_regex = '|'.join(bad_channel_names)
#     raw_removed.set_montage('standard_1020', on_missing='warn')
#     picks = mne.pick_channels_regexp(raw_removed.ch_names, regexp=bad_channels_regex)
#     raw_removed.plot(order=picks, n_channels=len(picks), scalings={'eeg': 1e-4})


# def ica_analysis(raw_removed):
#     ica = mne.preprocessing.ICA(n_components=10, random_state=42, max_iter="auto")
#     ica.fit(raw_removed)
#     ica.exclude = [0, 1, 2, 3, 4, 5, 6]  # details on how we picked these are omitted here
#     raw_reconstructed = raw_removed.copy()
#     ica.apply(raw_reconstructed)
#     return raw_reconstructed


# def ica_plot(raw_reconstructed):
#     # ica = mne.preprocessing.ICA(n_components=10, random_state=42, max_iter="auto")
#     # ica.fit(raw_removed)
#     # ica.exclude = [0,1,2,3,4,5,6,7,8,9]  # details on how we picked these are omitted here
#     # raw_reconstructed = raw_removed.copy()
#     ica.apply(raw_reconstructed)
#     ica.plot_sources(raw_removed, show_scrollbars=False)
#     ica.plot_components()
#     ica.plot_properties(raw_removed, picks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#     ica.plot_overlay(raw_removed, exclude=[0], picks="eeg")
#     raw_removed.plot_sensors(show_names=True)

#     explained_var_ratio = ica.get_explained_variance_ratio(raw_removed)
#     for channel_type, ratio in explained_var_ratio.items():
#         print(
#             f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
#         )
#     explained_var_ratio = ica.get_explained_variance_ratio(raw_removed, components=[0], ch_type="eeg")

#     # This time, print as percentage
#     ratio_percent = round(100 * explained_var_ratio["eeg"])
#     print(
#         f"Fraction of variance in EEG signal explained by first component: "
#         f"{ratio_percent}%")


# def create_dataset(raw_reconstructed, eeg_stream, marker_stream, df_marker):
#     # Sometimes Lab Recorder accidentally adds another empty stream to the xdf file, so we need to check and get rid of it
#     # we had 3 incoming channels: EE225-000000-000867-02-DESKTOP-, EE225-000000-000867-02-TRG-DESK, LSLMarkersInletStreamName2 (MSI)
#     # if len(data) > 3:
#     #     for i, _ in enumerate(data):
#     #         if 0 in data[i]["time_series"].shape:
#     #             data.pop(i)

#     # eeg_stream = find_stream('eeg', data)
#     # eeg_time_series = eeg_stream["time_series"] #EEG time series data.
#     eeg_time_series = raw_reconstructed.get_data()
#     eeg_time_series = eeg_time_series.T

#     # eeg_data = eeg_stream["time_series"].T
#     eeg_timestamps = eeg_stream["time_stamps"]  # Timestamps for each sample in the EEG time series.
#     # timeseries and stamps for all markers
#     event_time_series = marker_stream['time_series']
#     event_time_stamps = marker_stream['time_stamps']

#     event_time_series_onset = df_marker[df_marker['marker'].str.startswith('StimStart')].reset_index()
#     event_time_series_onset['numb'] = range(len(event_time_series_onset))
#     event_time_series_onset = event_time_series_onset.rename(columns={'time_stamps': 'start', 'start_time': 'start_time'})

#     event_time_series_offset = df_marker[df_marker['marker'].str.startswith('StimEnd')].reset_index()
#     event_time_series_offset['numb'] = range(len(event_time_series_offset))
#     event_time_series_offset = event_time_series_offset.rename(columns={'time_stamps': 'end', 'end_time': 'end_time'})

#     diff = np.subtract.outer(event_time_stamps, eeg_timestamps)
#     eeg_indices = np.argmin(np.abs(diff), axis=1)
#     eeg_indices_onset = eeg_indices[::2]  # every second index because we have markers for StimStart and StimEnd consecutivley
#     start = eeg_indices_onset[0]  # index at which first event aligns within the EEG
#     # Initialize an array to hold the labels for the EEG samples
#     labels = np.empty(eeg_timestamps.shape, dtype=int)
#     labels.fill(900)  # Fill with 900 (break)
#     group_labels = np.copy(labels)
#     trials = np.copy(labels)

#     # Grab the EEG data from first event onwards and turn into dataframe
#     eeg_data = eeg_time_series[start:, :64]

#     # montage = mne.channels.make_standard_montage('biosemi64')
#     # ch_names_set = montage.ch_names

#     ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1',
#                 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz',
#                 'P4', 'P8', 'POz', 'O1', 'O2', 'EOG', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2',
#                 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2',
#                 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz']
#     out = pd.DataFrame(eeg_data, columns=ch_names)

#     # Mapping for group labels
#     group_mapping = {
#         'banana': 1,
#         'strawberry': 2,
#         'panda': 3,
#         'basketball': 4,
#         'face': 5,
#         'tiger': 6
#     }

#     # Function to map group and specific labels to integers
#     def map_group_and_specific_labels(marker):
#         # Split the marker string and extract the relevant parts
#         parts = marker.split()
#         group_label_str = ''.join(filter(str.isalpha, parts[1]))  # Extract only the alphabetic part
#         specific_label_str = parts[1]

#         # Map group label
#         group_label = group_mapping.get(group_label_str, 0)

#         # Map specific label by adhering group label and numeric part as a string
#         if group_label > 0:
#             specific_number_str = ''.join(filter(str.isdigit, specific_label_str))
#             specific_label = int(f"{group_label}{specific_number_str}")  # e.g., banana1 -> 11, banana12 -> 112
#         else:
#             specific_label = 0

#         return group_label, specific_label

#     # Apply the mapping function to each row
#     df_marker[['group_label_int', 'specific_label_int']] = df_marker['marker'].apply(
#         lambda x: pd.Series(map_group_and_specific_labels(x))
#     )

#     # Extract the group and specific labels as strings for clarity
#     df_marker['group_label'] = df_marker['marker'].apply(lambda x: ''.join(filter(str.isalpha, x.split()[1])))
#     df_marker['specific_label'] = df_marker['marker'].apply(lambda x: x.split()[1])

#     df_marker_onset = df_marker[::2]

#     # Initialize a dictionary to keep track of the trial counts for each specific label
#     trial_counters = {}

#     # Function to assign trial labels with incremental counts for each occurrence of the same specific label
#     def assign_trial_label(row):
#         specific_label = row['specific_label_int']

#         # Increment the trial number every time we see the specific label
#         if specific_label in trial_counters:
#             trial_counters[specific_label] += 1
#         else:
#             trial_counters[specific_label] = 1

#         # Return the trial label with the current trial count
#         return int(f"{specific_label}{trial_counters[specific_label]}")

#     # Apply the trial label assignment to each row
#     df_marker_onset['trial_label'] = df_marker_onset.apply(assign_trial_label, axis=1)

#     # Create the final DataFrame with all relevant columns
#     df_int_label_onset = df_marker_onset[['marker', 'group_label', 'specific_label', 'group_label_int', 'specific_label_int', 'trial_label']]

#     # Iterate over the events
#     event_time = 8
#     fs = 512
#     # df_int_label_onset = df_int_label[::2]

#     for i in range(len(eeg_indices_onset)):
#         # Label all EEG samples + 2s after the event with the event's label, rest is Break (900)
#         event_end = int(eeg_indices_onset[i] + event_time * fs)
#         group_labels[eeg_indices_onset[i]:event_end] = int(df_int_label_onset.iloc[i]['group_label_int'])
#         labels[eeg_indices_onset[i]:event_end] = int(df_int_label_onset.iloc[i]['specific_label_int'])
#         trials[eeg_indices_onset[i]:event_end] = int(df_int_label_onset.iloc[i]['trial_label'])
#         # double check with haojun for the first three lines

#     dataset = []

#     df = pd.concat(
#         [pd.DataFrame({'time': eeg_timestamps[start:], 'label': labels[start:], 'group_label': group_labels[start:], 'trial_label': trials[start:]}), out],
#         axis=1)
#     df = df[:int(eeg_indices_onset[-1] + event_time * fs)]  # Remove the rest of the EEG samples that don't have a label

#     dataset.append(df)

#     return dataset


# def remove_breaks(dataset):
#     # remove the breaks
#     for i in range(len(dataset)):
#         dataset[i] = dataset[i][dataset[i]['group_label'] < 900]
#     return dataset


# def return_dataset(dataset, ch_names):
#     # #Double check with haojun
#     out = []
#     for df in dataset:
#         grouped = df.groupby('trial_label')  # maybe some error here???
#         trial_label = np.array([name for name, group in grouped])
#         labels = np.array([group['label'].iloc[0] for name, group in grouped])
#         group_labels = np.array([group['group_label'].iloc[0] for name, group in grouped])
#         eeg_matrix = np.array([group[ch_names].values for name, group in grouped])

#     out.append((trial_label, labels, group_labels, eeg_matrix))
#     return out


# def build_class_epochs_mne(out, sfreq, ch_names, bad_channels):
#     ## Build epochs for the 6 classes 
#     out_t = np.swapaxes(out[0][3], 1, 2)

#     events = np.column_stack((np.arange(468), np.zeros(468, dtype=int), np.array(out[0][2])))

#     event_dict = {'banana': 1,
#                   'strawberry': 2,
#                   'panda': 3,
#                   'basketball': 4,
#                   'face': 5,
#                   'tiger': 6}
#     eeg_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * 64)
#     eeg_epochs_classes = mne.EpochsArray(out_t, eeg_info, events=events, event_id=event_dict)
#     eeg_epochs_classes.info['bads'] = bad_channels
#     return eeg_epochs_classes


# def build_cat_epochs_mne(out, sfreq, ch_names, bad_channels):
#     ## Built Epochs for each category in the class
#     out_t = np.swapaxes(out[0][3], 1, 2)
#     events_cat = np.column_stack((np.arange(468), np.zeros(468, dtype=int), np.array(out[0][1])))
#     event_cat_id = {
#         'banana1': 11, 'banana2': 12, 'banana3': 13, 'banana4': 14, 'banana5': 15,
#         'banana6': 16, 'banana7': 17, 'banana8': 18, 'banana9': 19, 'banana10': 110,
#         'banana11': 111, 'banana12': 112, 'banana13': 113,
#         'strawberry1': 21, 'strawberry2': 22, 'strawberry3': 23, 'strawberry4': 24,
#         'strawberry5': 25, 'strawberry6': 26, 'strawberry7': 27, 'strawberry8': 28,
#         'strawberry9': 29, 'strawberry10': 210, 'strawberry11': 211, 'strawberry12': 212,
#         'strawberry13': 213,
#         'panda1': 31, 'panda2': 32, 'panda3': 33, 'panda4': 34, 'panda5': 35,
#         'panda6': 36, 'panda7': 37, 'panda8': 38, 'panda9': 39, 'panda10': 310,
#         'panda11': 311, 'panda12': 312, 'panda13': 313,
#         'basketball1': 41, 'basketball2': 42, 'basketball3': 43, 'basketball4': 44,
#         'basketball5': 45, 'basketball6': 46, 'basketball7': 47, 'basketball8': 48,
#         'basketball9': 49, 'basketball10': 410, 'basketball11': 411, 'basketball12': 412,
#         'basketball13': 413,
#         'face1': 51, 'face2': 52, 'face3': 53, 'face4': 54, 'face5': 55,
#         'face6': 56, 'face7': 57, 'face8': 58, 'face9': 59, 'face10': 510,
#         'face11': 511, 'face12': 512, 'face13': 513,
#         'tiger1': 61, 'tiger2': 62, 'tiger3': 63, 'tiger4': 64, 'tiger5': 65,
#         'tiger6': 66, 'tiger7': 67, 'tiger8': 68, 'tiger9': 69, 'tiger10': 610,
#         'tiger11': 611, 'tiger12': 612, 'tiger13': 613
#     }
#     eeg_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * 64)
#     eeg_epochs_objects = mne.EpochsArray(out_t, eeg_info, events=events_cat, event_id=event_cat_id)
#     eeg_epochs_objects.info['bads'] = bad_channels
#     return eeg_epochs_objects


# def plot_topo(eeg_epochs, eeg_info):
#     # epochs = mne.EpochsArray(eeg_epochs, eeg_info)
#     eeg_epochs.set_montage('standard_1020', on_missing='warn')
#     eeg_epochs.compute_psd().plot_topomap(ch_type='eeg', normalize=True)


# ## EEG data across time for each channel, where each channel's data is averaged across all epochs.
# def plot_eeg(eeg_epochs, eeg_info):
#     # epochs = mne.EpochsArray(eeg_epochs, eeg_info)
#     eeg_epochs.set_montage('standard_1020', on_missing='warn')
#     eeg_epochs.plot_image(combine='mean', picks='eeg')


# def plot_evoked(eeg_epochs, eeg_info, event_name):
#     """
#     Plot the evoked potential for a specific class.

#     Parameters:
#     -----------
#     eeg_epochs : mne.EpochsArray
#         The epochs array containing EEG data.
#     eeg_info : mne.Info
#         The info object containing metadata about the EEG recording.
#     event_name : str
#         The name of the event/class to plot the evoked potential for.
#     """
#     eeg_epochs.set_montage('standard_1020', on_missing='warn')
#     if event_name in eeg_epochs.event_id:
#         # Plot the average evoked potential for the specified class
#         evoked = eeg_epochs[event_name].average().crop(0, 1)
#         evoked.plot(picks="eeg", spatial_colors=True)
#     else:
#         print(f"Event '{event_name}' not found in event_id. Please check the event name.")


# def standartization(out):
#     out_t = np.swapaxes(out[0][3], 1, 2)
#     scaler = Scaler(scalings='mean')
#     eeg_epochs_standardized = scaler.fit_transform(out_t)
#     eeg_epochs_standardized.shape
#     return eeg_epochs_standardized


# def labelS02(event_time, subset_Stim_onset, eeg_timestamps, eeg_time_series):
#     ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1',
#                 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz',
#                 'P4', 'P8', 'POz', 'O1', 'O2', 'EOG', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2',
#                 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2',
#                 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz']
#     sub_event_time_stamps = subset_Stim_onset['index']
#     sub_event_time_stamps = np.array(sub_event_time_stamps)

#     diff = np.subtract.outer(sub_event_time_stamps, eeg_timestamps)
#     eeg_indices_onset = np.argmin(np.abs(diff), axis=1)
#     start = eeg_indices_onset[0]
#     end = eeg_indices_onset[-1]

#     eeg_sub = eeg_time_series[start:end, :64]

#     def assign_trial_label(row):
#         trial_counters = {}
#         specific_label = row['specific_label_int']

#         # Increment the trial number every time we see the specific label
#         if specific_label in trial_counters:
#             trial_counters[specific_label] += 1
#         else:
#             trial_counters[specific_label] = 1

#         # Return the trial label with the current trial count
#         return int(f"{specific_label}{trial_counters[specific_label]}")

#     # Apply the function to extract group and specific labels

#     out = pd.DataFrame(eeg_sub, columns=ch_names)
#     subset_Stim_onset['trial_label'] = subset_Stim_onset.apply(assign_trial_label, axis=1)
#     sub_df = subset_Stim_onset[['marker', 'group_label_int', 'specific_label_int', 'trial_label']]

#     """
#         - event_time: has to be set to the time of the specific event
#         - sub_marker : this is the subset of the marker dataframe for a specific event 
#         - out: stores the eeg_timeseries for the specific part and event and ch_names
#         - eeg_indices_onset: contain all the eeg_timestamps for onset of events
#         - eeg_timestamps: timestamps of eeg data
    
#     """

#     labels = np.empty(eeg_timestamps.shape, dtype=int)
#     labels.fill(900)  # Fill with 900 (break)
#     group_labels = np.copy(labels)
#     trials = np.copy(labels)
#     fs = 512
#     # df_int_label_onset = df_int_label[::2]
#     for i in range(len(eeg_indices_onset)):
#         # Label all EEG samples + 2s after the event with the event's label, rest is Break (900)
#         event_end = int(eeg_indices_onset[i] + event_time * fs)
#         group_labels[eeg_indices_onset[i]:event_end] = int(sub_df.iloc[i]['group_label_int'])
#         labels[eeg_indices_onset[i]:event_end] = int(sub_df.iloc[i]['specific_label_int'])
#         trials[eeg_indices_onset[i]:event_end] = int(sub_df.iloc[i]['trial_label'])
#         # double check with haojun for the first three lines

#     dataset = []

#     df = pd.concat([pd.DataFrame(
#         {'time': eeg_timestamps[eeg_indices_onset[0]:], 'label': labels[eeg_indices_onset[0]:], 'group_label': group_labels[eeg_indices_onset[0]:],
#          'trial_label': trials[eeg_indices_onset[0]:]}), out], axis=1)
#     df = df[:int(eeg_indices_onset[-1] + event_time * fs)]  # Remove the rest of the EEG samples that don't have a label
#     df = df.iloc[:len(out)]
#     dataset.append(df)

#     return dataset
