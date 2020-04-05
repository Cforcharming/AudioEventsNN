import xml.etree.ElementTree as ET
import scipy.io.wavfile as wavfile
import tensorflow as tf
import numpy as np


def load_data(db_level=None):
    """
    This module loads data from MIVIA dataset. For example:
        (x_train, y_train), (x_test, y_test) = mivia_db.load_data()
        
    :return x_train: tf.data.Dataset Training set, contains STFT format 19x513 tf.Tensor.
    :return y_train: tf.data.Dataset Training label, contains 1x1 label tf.Tensor.
    :return x_test: tf.data.Dataset Testing set, contains STFT format 19x513 tf.Tensor.
    :return y_test: tf.data.Dataset Testing label, contains 1x1 label tf.Tensor.
    """
    (x_train_waves, y_train_names), (x_test_waves, y_test_names), (x_train_lens, x_test_lens) = _gets(db_level)
    
    def _train_gen():
        """
        Generator of x_train dataset, yields a single STFT format 19x513 tf.Tensor.
        """
        for (wave, name, length) in zip(x_train_waves, y_train_names, x_train_lens):
            temp_set1 = _get_stfts_from_wave(wave)
            temp_set2 = _get_labels_from_file(name, length)
            for (i, j) in zip(temp_set1, temp_set2):
                yield [i.numpy(), j.numpy()]
        
    def _y_train_gen():
        """
        Generator of y_train dataset, yields a single 1x1 label tf.Tensor.
        """
        for (name, length) in zip(y_train_names, x_train_lens):
            temp_set = _get_labels_from_file(name, length)
            for i in temp_set:
                yield i
    
    def _test_gen():
        """
        Generator of x_test dataset, yields a single STFT format 19x513 tf.Tensor.
        """
        for (wave, name, length) in zip(x_test_waves, y_test_names, x_test_lens):
            temp_set1 = _get_stfts_from_wave(wave)
            temp_set2 = _get_labels_from_file(name, length)
            for (i, j) in zip(temp_set1, temp_set2):
                yield tuple([i.numpy(), j.numpy()])
    
    def _y_test_gen():
        """
        Generator of y_test dataset, yields a single 1x1 label tf.Tensor.
        """
        for (name, length) in zip(y_test_names, x_test_lens):
            temp_set = _get_labels_from_file(name, length)
            for i in temp_set:
                yield i
    
    train = tf.data.Dataset.from_generator(_train_gen, output_types=tf.complex64)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    # y_train = tf.data.Dataset.from_generator(_y_train_gen, output_types=tf.complex64)\
    #     .prefetch(tf.data.experimental.AUTOTUNE)
    test = tf.data.Dataset.from_generator(_test_gen, output_types=tf.complex64)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    # y_test = tf.data.Dataset.from_generator(_y_test_gen, output_types=tf.complex64)\
    #     .prefetch(tf.data.experimental.AUTOTUNE)
    #
    return train, test


def _gets(db_level=None):
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

    x_train_waves, y_train_names, x_test_waves, y_test_names, x_train_lens, x_test_lens = [], [], [], [], [], []
    for db_path in db_paths:
        for i in range(1, 2):  # training set contains 66 files
            ind_audio_path = '%s%02d%s' % (gen_audio_paths[0], i, db_path)
            ind_xml_path = '%s%02d.xml' % (gen_xml_paths[0], i)
            fs, wave = wavfile.read(ind_audio_path)
            x_train_waves.append(wave)
            x_train_lens.append(len(wave))
            y_train_names.append(ind_xml_path)
        
        for i in range(1, 2):  # testing set contains 29 files
            ind_audio_path = '%s%02d%s' % (gen_audio_paths[1], i, db_path)
            ind_xml_path = '%s%02d.xml' % (gen_xml_paths[1], i)
            fs, wave = wavfile.read(ind_audio_path)
            x_test_waves.append(wave)
            x_test_lens.append(len(wave))
            y_test_names.append(ind_xml_path)
    
    return (x_train_waves, y_train_names), (x_test_waves, y_test_names), (x_train_lens, x_test_lens)


def _get_stfts_from_wave(wave):
    """
    Convert a wave from int16 to -1.~+1. float32,
    split it into 10240 points(320ms) slices,
    then transform the slice by STFT, with FFT length 1024(32ms), 50% overlap.
    Generate a sub dataset from a single file
    :param wave: np.ndarray The wave of the audio of one .wav file.
    :return temp_set: tf.data.Dataset The dataset of the STFT slices(19x513) of this particular wave.
    """
    stfts = tf.signal.stft(wave.astype('float32')/32768., frame_length=1024, frame_step=512, pad_end=True)
    temp_set = tf.data.Dataset.from_tensor_slices(stfts).batch(19, drop_remainder=True)
    return temp_set.prefetch(tf.data.experimental.AUTOTUNE)


def _get_labels_from_file(file_path, x_lens):
    """
    Generate a sub dataset from a single file.
    :param file_path:str The file path of one .xml label file.
    :param x_lens:int The length of the .wav file for the .xml to label
    :return temp_set: tf.data.Dataset The dataset of 1x1 label of one particular .wav file.
    """
    root = ET.parse(file_path).getroot()
    classes, start, end = [], [], []
    for (class_id, start_t, end_t) in zip(
            root.findall('./events/item/CLASS_ID'),
            root.findall('./events/item/STARTSECOND'),
            root.findall('./events/item/ENDSECOND')):
        classes.append(float(class_id.text))
        start.append(float(start_t.text))
        end.append(float(end_t.text))
        
    classes = np.array(classes)
    start_frame = (np.array(start) * 3.125).astype('int')
    end_frame = (np.array(end) * 3.125).astype('int')
    frame_len = x_lens // 10240
    y = np.zeros([frame_len, 1, 4])
    
    for i in range(len(classes)):
        for j in range(start_frame[i], end_frame[i]+1):
            y[j][classes[i]] = 1
    
    temp_set = tf.data.Dataset.from_tensor_slices(y)
    return temp_set.prefetch(tf.data.experimental.AUTOTUNE)
