import mne
import matplotlib.pyplot as plt
%matplotlib qt

# participant selection
participant_id = '01'  # 01, 04, 05, 06, 07, 09, 11, 12, 13, 14
file_path = f'OpenMIIR-RawEEG_v1/P{participant_id}-raw.fif'

# load data
raw = mne.io.read_raw_fif(file_path, preload=True)

# print keys
print(raw.info.keys())

# show the EEG
raw.plot(title=f"EEG Data for Participant {participant_id}")
plt.show()

# show the general information
print(raw.info)

# events
events = mne.find_events(raw)
print(events)
trial_events = mne.find_events(raw, stim_channel='STI 014', shortest_event=0)
plt.title(f"Event Plot for Participant {participant_id}")

# plot enevts            
plt.figure(figsize=(17, 10))
axes = plt.gca()
mne.viz.plot_events(trial_events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, axes=axes)

# print the time point of the first and the last event
print('First event at ', raw.times[trial_events[0, 0]])
print('Last event at ', raw.times[trial_events[-1, 0]])
plt.show()