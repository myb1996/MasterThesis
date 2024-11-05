import mne
import matplotlib.pyplot as plot
%matplotlib qt

# participant selection
participant_id = '01'  # 01, 04, 05, 06, 07, 09, 11, 12, 13, 14
file_path = f'OpenMIIR-RawEEG_v1/P{participant_id}-raw.fif'

# load data
raw = mne.io.read_raw_fif(file_path, preload=True)

# plot sensors
raw.plot_sensors(show_names=True)

# set montage
montage = mne.channels.make_standard_montage('biosemi64')
raw.set_montage(montage)

# check 3D montage
raw.plot_sensors(kind='topomap', show_names=True)
fig = mne.viz.plot_alignment(raw.info, trans=None, show_axes=True)
mne.viz.set_3d_view(fig, azimuth=180, elevation=90)

# interpolate bad channels
raw.load_data()
raw.interpolate_bads()
raw.info['bads']

# check if bad channels have been fixed
raw.set_eeg_reference()
raw.plot_sensors(show_names=True)

# PSD before and after filter
raw.plot_psd()
raw.filter(l_freq=0.5, h_freq=35)
raw.plot_psd()