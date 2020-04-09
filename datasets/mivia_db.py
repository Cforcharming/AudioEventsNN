import xml.etree.ElementTree as ET
import scipy.io.wavfile as wavfile
import tensorflow as tf
import numpy as np


def load_data(db_level=None, one_hot=False):
    """
    This module loads data from MIVIA dataset. For example:
        train, test = mivia_db.load_data()
        
    :return train: yields (x_train, y_train), where x_train is 128x128
    :return train: yields (x_test, y_test), where x_train is 128x128
    """
#     (x_train_waves, y_train_names), (x_test_waves, y_test_names), (x_train_lens, x_test_lens) = _gets(db_level)
#
#     # def _x_train_gen():
#     #     """
#     #     Generator of x_train dataset, yields a single STFT format 128x128 tf.Tensor.
#     #     """
#     #     for wave in x_train_waves:
#     #         temp_set = _get_mel_logs_from_wave(wave)
#     #         for i in temp_set:
#     #             yield i
#
#     def _y_train_gen():
#         """
#         Generator of y_test dataset, yields a single 1x1 label tf.Tensor.
#         """
#         for (wave, name, length) in zip(x_train_waves, y_train_names, x_train_lens):
#             temp_set1 = _get_mel_logs_from_wave(wave)
#             temp_set2 = _get_labels_from_file(name, length, one_hot)
#             for (i, j) in zip(temp_set1, temp_set2):
#                 print('in train gen', i.shape, j.shape)
#                 yield np.array(i), j
#
#     # def _x_test_gen():
#     #     """
#     #     Generator of x_test dataset, yields a single STFT format 128x128 tf.Tensor.
#     #     """
#     #     for wave in x_test_waves:
#     #         temp_set = _get_mel_logs_from_wave(wave)
#     #         for i in temp_set:
#     #             yield i
#
#     def _y_test_gen():
#         """
#         Generator of y_test dataset, yields a single 1x1 label tf.Tensor.
#         """
#         for (wave, name, length) in zip(x_test_waves, y_test_names, x_test_lens):
#             temp_set1 = _get_mel_logs_from_wave(wave)
#             temp_set2 = _get_labels_from_file(name, length, one_hot)
#             for (i, j) in zip(temp_set1, temp_set2):
#                 print('in test gen', i.shape, j.shape)
#                 yield np.array(i), j
#
#     # x_train = tf.data.Dataset.from_generator(_x_train_gen, output_types=tf.float32)
#     y_train = tf.data.Dataset.from_generator(_y_train_gen, output_types=tf.float32)
#     # x_test = tf.data.Dataset.from_generator(_x_test_gen, output_types=tf.float32)
#     y_test = tf.data.Dataset.from_generator(_y_test_gen, output_types=tf.float32)
#     return y_train, y_test
#     return tf.data.Dataset.from_tensor_slices((x_train, y_train)), tf.data.Dataset.from_tensor_slices((x_test,y_test))
#
#
# def _gets(db_level=None):
    """
    Generate file paths for .wav and .xml files from the dataset.
    Then get the wave data and its length from the generated .wav file.
    
    :return x_train_waves: list<np.ndarray> List of training set waves of .wav files.
    :return y_train_names: list<str> List of training set .xml label files.
    :return x_test_waves: list<np.ndarray> List of testing set waves.
    :return y_test_names: list<str> List of testing set .xml label files.
    :return x_train_lens: list<int> List of lengths of waves in x_train_waves.
    :return x_test_lens: list<int> List of lengths of waves in x_test_waves.
    """
    gen_audio_paths = ('data/MIVIA_DB/training/sounds/000', 'data/MIVIA_DB/testing/sounds/000')
    gen_xml_paths = ('data/MIVIA_DB/training/000', 'data/MIVIA_DB/testing/000')
    
    # each number in db_paths shows no difference in labels but the SNR(dB): 5dB, 10dB, 15dB, 20dB, 25dB, 30dB
    db_paths = ('_1.wav', '_2.wav', '_3.wav', '_4.wav', '_5.wav', '_6.wav')
    if db_level == 5:
        db_paths = [db_paths[0]]
    elif db_level == 10:
        db_paths = [db_paths[1]]
    elif db_level == 15:
        db_paths = [db_paths[2]]
    elif db_level == 20:
        db_paths = [db_paths[3]]
    elif db_level == 25:
        db_paths = [db_paths[4]]
    elif db_level == 30:
        db_paths = [db_paths[5]]
    
    # x_train_waves, y_train_names, x_test_waves, y_test_names, x_train_lens, x_test_lens = [], [], [], [], [], []
    
    def _train_gen():
        for db_path in db_paths:
            for i in range(1, 3):  # training set contains 66 files
                ind_audio_path = '%s%02d%s' % (gen_audio_paths[0], i, db_path)
                ind_xml_path = '%s%02d.xml' % (gen_xml_paths[0], i)
                fs, wave = wavfile.read(ind_audio_path)
                log_mels = _get_mel_logs_from_wave(wave)
                labels = _get_labels_from_file(ind_xml_path, len(wave), one_hot)
                for (log_mel, label) in zip(log_mels, labels):
                    yield tf.reshape(log_mel, [1, 128, 128]), label
    
    def _test_gen():
        for db_path in db_paths:
            for i in range(1, 3):  # testing set contains 29 files
                ind_audio_path = '%s%02d%s' % (gen_audio_paths[1], i, db_path)
                ind_xml_path = '%s%02d.xml' % (gen_xml_paths[1], i)
                fs, wave = wavfile.read(ind_audio_path)
                log_mels = _get_mel_logs_from_wave(wave)
                labels = _get_labels_from_file(ind_xml_path, len(wave), one_hot)
                for (log_mel, label) in zip(log_mels, labels):
                    yield tf.reshape(log_mel, [1, 128, 128]), label
    
    # return (x_train_waves, y_train_names), (x_test_waves, y_test_names), (x_train_lens, x_test_lens)
    train_set = tf.data.Dataset.from_generator(generator=_train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=(tf.TensorShape((1, 128, 128)), tf.TensorShape((1, )))
                                               ).batch(32)
    test_set = tf.data.Dataset.from_generator(generator=_test_gen,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape((1, 128, 128)), tf.TensorShape((1, )))
                                              ).batch(32)
    
    return train_set, test_set


def _get_mel_logs_from_wave(wave):
    """
    Convert a wave from int16 to -1.~+1. float32,\]
    split it into 16384 points(512ms) slices,
    then transform the slice by STFT, with FFT length 256(8ms), 50% overlap.
    Generate a sub dataset from a single file
    Args:
        wave: 'np.array` The wave of the audio of one .wav file.
    Returns:
         temp_set: `tf.data.Dataset` The dataset of the STFT slices(19x513) of this particular wave.
    """
    stfts = tf.signal.stft(
        wave.astype('float32') / 32768.,
        frame_length=256,
        frame_step=128,
        pad_end=True
    )
    spectrograms = tf.abs(stfts)
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=129,
        sample_rate=32000,
        lower_edge_hertz=80.,
        upper_edge_hertz=7600.
    )
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    log_mel = tf.signal.frame(
        signal=tf.reshape(log_mel_spectrograms, [-1]),
        frame_length=16384,
        frame_step=16384,
        pad_end=False
    )
    return log_mel


def _get_labels_from_file(file_path, x_lens, one_hot):
    """
    Generate a sub dataset from a single file.
    Args:
        file_path: `str` The file path of one .xml label file.
        x_lens: `int` The length of the .wav file for the .xml to label
    Returns:
        temp_set: tf.data.Dataset The dataset of 1x1 label of one particular .wav file.
    """
    root = ET.parse(file_path).getroot()
    classes, start, end = [], [], []
    for (class_id, start_t, end_t) in zip(
            root.findall('./events/item/CLASS_ID'),
            root.findall('./events/item/STARTSECOND'),
            root.findall('./events/item/ENDSECOND')
    ):
        classes.append(float(class_id.text))
        start.append(float(start_t.text))
        end.append(float(end_t.text))
    
    classes = np.array(classes, dtype='float32')
    start_point = (np.array(start) * 32000).astype('int')
    end_point = (np.array(end) * 32000).astype('int')
    events = np.ones(x_lens)
    for i in range(len(classes)):
        for j in range(start_point[i], end_point[i] + 1):
            events[j] = classes[i]
    events_frame = tf.signal.frame(
        signal=events,
        frame_length=256,
        frame_step=128,
        pad_end=True,
        pad_value=1.
    )
    y = tf.reduce_max(events_frame, axis=1)
    y_frame = tf.signal.frame(
        signal=y,
        frame_length=128,
        frame_step=128,
        pad_end=False
    )
    y_label = tf.reduce_max(y_frame, axis=1)
    if one_hot:
        # TODO fix tensorflow.python.framework.errors_impl.NotFoundError: Could not find valid device for node.
        yl = tf.one_hot(y_label.numpy().astype('int')-1, 4)
    else:
        yl = tf.expand_dims(y_label, axis=1)
    return yl
