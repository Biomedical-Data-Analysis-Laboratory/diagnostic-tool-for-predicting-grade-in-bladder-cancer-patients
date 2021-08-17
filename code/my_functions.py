# region IMPORTS
# -------------------  My files  ---------------------
import my_constants
# -------------------  GPU Stuff  ---------------------
import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.models import *
from keras.layers import *
from keras.callbacks import CSVLogger
# -------------------  Other  ---------------------
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt
import itertools
import datetime
import logging
my_logger = logging.getLogger('data_logger')
import time
import pickle
import os
import pyvips
# endregion


def init_file(SAVED_DATA_FOLDER, LOG_FOLDER, START_NEW_MODEL, CONTINUE_FROM_MODEL, METADATA_FOLDER, USE_MULTIPROCESSING):
    # Function that runs in the beginning of the program
    if START_NEW_MODEL in [True, 'True', 'true']:
        # Start a new model
        my_print('Starting new project')

        # Make a new folder to save current run inside
        current_run_path = '{}{}'.format(SAVED_DATA_FOLDER, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S/'))
        os.makedirs(current_run_path, exist_ok=True)

    elif START_NEW_MODEL in [False, 'False', 'false']:
        # Check to see that previous projects exists
        if os.listdir(SAVED_DATA_FOLDER).__len__() == 0:
            my_print('No previous projects found. Please start a new project by setting START_NEW_MODEL to True. Stopping program.')
            exit()
        else:
            my_print('Continue previous project')

        if CONTINUE_FROM_MODEL == 'last':
            project_list = os.listdir(SAVED_DATA_FOLDER)
            project_list.sort()
            current_run_path = '{}{}'.format(SAVED_DATA_FOLDER, project_list[-1] + '/')
        elif CONTINUE_FROM_MODEL in os.listdir(SAVED_DATA_FOLDER):
            current_run_path = '{}{}/'.format(SAVED_DATA_FOLDER, CONTINUE_FROM_MODEL)
        else:
            my_print('Project specified in CONTINUE_FROM_MODEL not found. Stopping program.')
            exit()
    else:
        my_print('Wrong format of START_NEW_MODEL. Please choose True or False. Stopping program.')
        exit()

    # Create a new summary.csv file, A CSV file that includes summary of all model modes. Can be opened in Excel.'
    SUMMARY_CSV_FILE_PATH = current_run_path + 'Summary_training.csv'
    summary_csv_file_create_new(SUMMARY_CSV_FILE_PATH=SUMMARY_CSV_FILE_PATH)

    # Check if SAVED_DATA_FOLDER exist. If not, create one
    os.makedirs(SAVED_DATA_FOLDER, exist_ok=True)

    # Create metadata folder
    os.makedirs(current_run_path + METADATA_FOLDER, exist_ok=True)

    # Make a new folder for logs
    current_log_path = '{0}{1}'.format(current_run_path, LOG_FOLDER)
    os.makedirs(current_log_path, exist_ok=True)

    # Test start time
    global start_time
    start_time = time.time()
    start_time_formatted = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    start_time_logger = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H-%M-%S')

    # Create a logger (logger is created in top of file (import section))
    my_logger_path = '{0}{1}.log'.format(current_log_path, start_time_logger)
    my_logger.setLevel(level=logging.DEBUG)
    fh = logging.FileHandler(my_logger_path)
    my_logger.addHandler(fh)

    if USE_MULTIPROCESSING is False:
        my_print('Warning: MULTIPROCESSING IS DISABLED!')

    # Print test start
    my_print('Program started at {}'.format(start_time_formatted))

    return current_run_path, SUMMARY_CSV_FILE_PATH


def my_print(msg, visible=True, error=False):
    # Function that both prints a message to console and to a log file
    if not error:
        my_logger.info(msg)
        if visible:
            print(msg)
    else:
        msg = 'ERROR: {}'.format(msg)
        my_logger.error(msg)
        print(msg)


def create_keras_logger(summary_path, model_name):
    # Create a log to save epoches/accuracy/loss
    log_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
    log_name = '{}{}keras_log_{}.csv'.format(summary_path, model_name, log_timestamp)
    csv_logger = CSVLogger(log_name, append=True, separator=';')
    return csv_logger


def remove_white_background_v3(input_img, PADDING, folder_path):
    # Reset variables
    remove_rows_top = 0
    remove_rows_bottom = 0
    remove_cols_left = 0
    remove_cols_right = 0
    x_list = []
    y_list = []
    white_background_vector = [250, 251, 252, 253, 254, 255]
    csv_override_filename = folder_path + 'override.csv'

    if os.path.isfile(csv_override_filename):
        # Some images need special care. We can override the values of x_inside and y_inside here using the CSV file in the folder
        # csv file in the folder should have the name 'override.csv', and contain values "x,y", e.g. "1000,2500".
        # Read from CSV file
        with open(csv_override_filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for line in reader:
                override_xy = line

        if input_img.height < 25000:
            x_inside = int(override_xy[0])
            y_inside = int(override_xy[1])
        elif input_img.height < 50000:
            x_inside = int(override_xy[0]) // my_constants.Scale_between_25x_100x
            y_inside = int(override_xy[1]) // my_constants.Scale_between_25x_100x
        elif input_img.height > 100000:
            x_inside = int(override_xy[0]) // my_constants.Scale_between_25x_400x
            y_inside = int(override_xy[1]) // my_constants.Scale_between_25x_400x
    else:
        # If there is not a CSV file with coordinates of x_inside and y_inside, this section will find them automatically.
        # Make a grid of all X lines and find minimum values (which indicate a WSI in the large white space) (there may be more than one WSI)
        step_x = int(input_img.width // 100)
        step_y = int(input_img.height // 100)

        for x_pos in range(step_x, input_img.width, step_x):
            tmp = input_img.extract_area(x_pos, 0, 1, input_img.height)
            x_list.append((x_pos, tmp.min()))

        # Go through x_list and find all transitions between "white background" and "WSI image".
        threshold = 250
        dict_of_transitions_x = dict()
        over_under_threshold = 'under'

        for index, value in enumerate(x_list):
            if over_under_threshold == 'under':
                if value[1] < threshold:
                    dict_of_transitions_x[len(dict_of_transitions_x)] = index
                    over_under_threshold = 'over'
            elif over_under_threshold == 'over':
                if value[1] > threshold:
                    dict_of_transitions_x[len(dict_of_transitions_x)] = index
                    over_under_threshold = 'under'

        x_inside = x_list[dict_of_transitions_x[0]][0] + ((x_list[dict_of_transitions_x[1]][0] - x_list[dict_of_transitions_x[0]][0]) // 2)

        # Initial crop (if there are more than one WSI in the image, this crops out the first one)
        if len(dict_of_transitions_x) > 2:
            init_crop_x = x_list[dict_of_transitions_x[1]][0] + ((x_list[dict_of_transitions_x[2]][0] - x_list[dict_of_transitions_x[1]][0]) // 2)
            input_img = input_img.extract_area(0, 0, init_crop_x, input_img.height)

        # Make a grid of all Y lines and find minimum values (which indicate a WSI in the large white space)
        for y_pos in range(step_y, input_img.height, step_y):
            tmp = input_img.extract_area(0, y_pos, input_img.width, 1)
            y_list.append((y_pos, tmp.min()))

        dict_of_transitions_y = dict()
        over_under_threshold = 'under'

        for index, value in enumerate(y_list):
            if over_under_threshold == 'under':
                if value[1] < threshold:
                    dict_of_transitions_y[len(dict_of_transitions_y)] = index
                    over_under_threshold = 'over'
            elif over_under_threshold == 'over':
                if value[1] > threshold:
                    dict_of_transitions_y[len(dict_of_transitions_y)] = index
                    over_under_threshold = 'under'

        y_inside = y_list[dict_of_transitions_y[0]][0] + ((y_list[dict_of_transitions_y[1]][0] - y_list[dict_of_transitions_y[0]][0]) // 2)

    ##### REMOVE HORIZONTAL WHITE LINES (TOP AND DOWN)
    if input_img(x_inside, 0)[1] in white_background_vector:
        first = 0
        last = y_inside
        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            if input_img(x_inside, midpoint)[1] in white_background_vector:
                first = midpoint + 1
            else:
                last = midpoint - 1
        remove_rows_top = midpoint - 1
    ##### REMOVE HORIZONTAL WHITE LINES (BOTTOM AND UP)
    if input_img(x_inside, (input_img.height - 1))[1] in white_background_vector:
        # first = (current_image.height // 2) - 5000
        first = y_inside
        last = input_img.height

        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            # if current_image(((current_image.width // current_divide_constant)-(current_image.width//4)), midpoint)[1] == 255:
            if input_img(x_inside, midpoint)[1] in white_background_vector:
                last = midpoint - 1
            else:
                first = midpoint + 1

        remove_rows_bottom = midpoint
    ##### REMOVE VERTICAL WHITE LINES (VENSTRE MOT HoYRE)
    if input_img(0, y_inside)[1] == 255:
        first = 0
        last = x_inside

        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            if input_img(midpoint, y_inside)[1] == 255:
                first = midpoint + 1
            else:
                last = midpoint - 1
        remove_cols_left = midpoint - 1
    ##### REMOVE VERTICAL WHITE LINES (HOYRE MOT VENSTRE)
    if input_img(input_img.width - 1, y_inside)[1] == 255:
        first = x_inside
        last = input_img.width
        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            if input_img(midpoint, y_inside)[1] == 255:
                last = midpoint - 1
            else:
                first = midpoint + 1
        remove_cols_right = midpoint + 1

    # Calculate new width/height of image and crop.
    if remove_rows_bottom != 0:
        # Calculate new width/height
        new_width = (input_img.width - remove_cols_left - (input_img.width - remove_cols_right))
        new_height = (input_img.height - remove_rows_top - (input_img.height - remove_rows_bottom))

        # Include a border around image (to extract 25x tiles later)
        remove_cols_left = remove_cols_left - PADDING
        remove_rows_top = remove_rows_top - PADDING
        new_width = new_width + 2 * PADDING
        new_height = new_height + 2 * PADDING

        return remove_cols_left, remove_rows_top, new_width, new_height


def pickle_load(path):
    with open(path, 'rb') as handle:
        output = pickle.load(handle)
    return output


def pickle_save(variable_to_save, path):
    with open(path, 'wb') as handle:
        pickle.dump(variable_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def summary_csv_file_create_new(SUMMARY_CSV_FILE_PATH):
    # Create a new summary.csv file
    if not os.path.isfile(SUMMARY_CSV_FILE_PATH):
        try:
            with open(SUMMARY_CSV_FILE_PATH, 'w') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(
                    ['Mode', 'Model name', 'Label',
                     'Base model', 'Freeze', 'blocks_to_unfreeze_vgg16_vgg19', 'delayed_unfreeze_start_epoche_vgg16_vgg19', 'Base model pooling',
                     'Training samples', 'Validation samples', 'Test samples', 'Augment classes', 'Augment multiplier', 'Tile size',
                     'Layer config', 'Learning rate', 'Batch size', 'Dropout',
                     'N_neurons1', 'N_neurons2', 'N_neurons3', 'EARLY_STOPPING_PATIENCE',
                     'Best train loss', 'Best train acc', 'Best val loss', 'Best val acc',
                     'Best val loss epoch', 'Best val acc epoch', 'Trained epoches', 'Total epochs',
                     'Latent size', 'Compression', 'Time(H:M:S)',
                     'Optimizer', 'ReduceLROnPlateau',
                     'Trainable params(Start)', 'Non-trainable params(Start)', 'Trainable params(End)', 'Non-trainable params(End)',
                     'Python', 'Keras', 'TensorFlow', 'Date'])
        except Exception as e:
            my_print('Error writing to file', error=True)
            my_print(e, error=True)


def summary_csv_file_update(SUMMARY_CSV_FILE_PATH, MODE, model_name, label,
                            base_model, freeze_base_model, blocks_to_unfreeze_vgg16_vgg19, delayed_unfreeze_start_epoche_vgg16_vgg19, base_model_pooling,
                            training_samples, validation_samples, test_samples, augment_classes, augment_multiplier, tile_size,
                            layer_config, learning_rate, batch_size, dropout,
                            n_neurons1, n_neurons2, n_neurons3, EARLY_STOPPING_PATIENCE,
                            best_train_loss, best_train_acc, best_val_loss, best_val_acc,
                            best_val_loss_epoch, best_val_acc_epoch, trained_epoches, total_epochs,
                            latent_size, compression, model_time,
                            optimizer, ReduceLRstatus,
                            n_trainable_parameters_start, n_non_trainable_parameters_start, n_trainable_parameters_end, n_non_trainable_parameters_end,
                            python_version, keras_version, tf_version):
    # Update existing summary.csv file
    try:
        with open(SUMMARY_CSV_FILE_PATH, 'a') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                [MODE, model_name, label,
                 base_model, freeze_base_model, blocks_to_unfreeze_vgg16_vgg19, delayed_unfreeze_start_epoche_vgg16_vgg19, base_model_pooling,
                 training_samples, validation_samples, test_samples, augment_classes, augment_multiplier, tile_size,
                 layer_config, str(learning_rate).replace('.', ','), batch_size, str(dropout).replace('.', ','),
                 n_neurons1, n_neurons2, n_neurons3, EARLY_STOPPING_PATIENCE,
                 str(best_train_loss).replace('.', ','), str(best_train_acc).replace('.', ','),
                 str(best_val_loss).replace('.', ','), str(best_val_acc).replace('.', ','),
                 best_val_loss_epoch, best_val_acc_epoch, trained_epoches, total_epochs,
                 latent_size, str(compression).replace('.', ','), model_time,
                 optimizer, ReduceLRstatus,
                 n_trainable_parameters_start, n_non_trainable_parameters_start, n_trainable_parameters_end, n_non_trainable_parameters_end,
                 '\'' + python_version, '\'' + keras_version, tf_version,
                 datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')])
    except Exception as e:
        my_print('Error writing to file', error=True)
        my_print(e, error=True)


def augment_tiles(aug_argument, tile):
    # Augment tiles by rotation and flipping
    if aug_argument == '1':
        return tile
    elif aug_argument == '2':
        # rot90
        tile = tile.rot(1)
    elif aug_argument == '3':
        # rot180
        tile = tile.rot(2)
    elif aug_argument == '4':
        # rot270
        tile = tile.rot(3)
    elif aug_argument == '5':
        # rot90_flipHoriz
        tile = tile.rot(1)
        tile = tile.flip(0)
    elif aug_argument == '6':
        # rot270_flipHoriz
        tile = tile.rot(3)
        tile = tile.flip(0)
    elif aug_argument == '7':
        # flipVert
        tile = tile.flip(1)
    elif aug_argument == '8':
        # rot180_flipVert
        tile = tile.rot(2)
        tile = tile.flip(1)
    return tile


def dataset_preprocessing(preprocess_mode, input_tile):
    if preprocess_mode == 0:
        return input_tile
    elif preprocess_mode == 1:
        # Normalize from 0-255 to 0-1. (PyVips)
        input_tile /= 255.0
        return input_tile
    elif preprocess_mode == 2:
        # 'RGB'->'BGR'. (Numpy)
        assert isinstance(input_tile, (np.ndarray, np.generic)), "Wrong datatype in dataset_preprocessing(). Using PyVips datatype, but need Numpy."
        input_tile = input_tile[..., ::-1]
        return input_tile
    elif preprocess_mode == 3:
        assert isinstance(input_tile, (np.ndarray, np.generic)), "Wrong datatype in dataset_preprocessing(). Using PyVips datatype, but need Numpy."

        # Subtrackt bladder cancer dataset mean. Values taken from Dataset E Train/Val, not test-set. (Numpy)
        mean = [158.7236, 115.7552, 153.6722]
        input_tile = input_tile.astype('float64', copy=False)
        input_tile[..., 0] -= mean[0]
        input_tile[..., 1] -= mean[1]
        input_tile[..., 2] -= mean[2]
        return input_tile


def list_of_CVTL_models(LAYER_CONFIG, BASE_MODEL, OPTIMIZER, LEARNING_RATE, N_NEURONS_FIRST_LAYER,
                        N_NEURONS_SECOND_LAYER, N_NEURONS_THIRD_LAYER, DROPOUT, freeze_base_model,
                        BASE_MODEL_POOLING, which_model_mode_to_use, which_scale_to_use_mono, which_scale_to_use_di,
                        augmentation_multiplier, WHAT_LABELS_TO_USE, middle_layer_config,
                        n_neurons_mid_first_layer, n_neurons_mid_second_layer):
    temp_model_dict = dict()
    TL_MODEL_PARAMETERS = []
    TL_MODELS_AND_LOSS_ARRAY = dict()

    # Create list of all classifier models
    n = 0
    for layer in LAYER_CONFIG:
        for mid_layer in middle_layer_config:
            for base_model in BASE_MODEL:
                for freeze in freeze_base_model:
                    for base_model_pooling in BASE_MODEL_POOLING:
                        for optimizer in OPTIMIZER:
                            for lr in LEARNING_RATE:
                                for n_mid_neurons1 in n_neurons_mid_first_layer:
                                    for n_mid_neurons2 in n_neurons_mid_second_layer:
                                        for n_neurons1 in N_NEURONS_FIRST_LAYER:
                                            for n_neurons2 in N_NEURONS_SECOND_LAYER:
                                                for n_neurons3 in N_NEURONS_THIRD_LAYER:
                                                    for dropout in DROPOUT:
                                                        for aug_mult in augmentation_multiplier:
                                                            for label in WHAT_LABELS_TO_USE:
                                                                if 'mono' in which_model_mode_to_use:
                                                                    for scale_in_mono in which_scale_to_use_mono:
                                                                        temp_model_dict.update(dict(ID=n, layer_config=layer, base_model=base_model, optimizer=optimizer, learning_rate=lr,
                                                                                                    n_neurons1=n_neurons1, n_neurons2=n_neurons2, n_neurons3=n_neurons3, dropout=dropout,
                                                                                                    freeze_base_model=freeze, base_model_pooling=base_model_pooling,
                                                                                                    which_scale_to_use=scale_in_mono, trained_epoches=0, early_stopping=0,
                                                                                                    augment_multiplier=aug_mult, model_mode='mono', model_trained_flag=0, fold_trained_flag=0,
                                                                                                    what_labels_to_use=label, mid_layer_config=mid_layer, n_mid_neurons1=n_mid_neurons1,
                                                                                                    n_mid_neurons2=n_mid_neurons2))
                                                                        TL_MODEL_PARAMETERS.append(temp_model_dict.copy())
                                                                        n = n + 1

                                                                if 'di' in which_model_mode_to_use:
                                                                    for scale_in_di in which_scale_to_use_di:
                                                                        temp_model_dict.update(dict(ID=n, layer_config=layer, base_model=base_model, optimizer=optimizer, learning_rate=lr,
                                                                                                    n_neurons1=n_neurons1, n_neurons2=n_neurons2, n_neurons3=n_neurons3, dropout=dropout,
                                                                                                    freeze_base_model=freeze, base_model_pooling=base_model_pooling,
                                                                                                    which_scale_to_use=scale_in_di, trained_epoches=0, early_stopping=0,
                                                                                                    augment_multiplier=aug_mult, model_mode='di', model_trained_flag=0, fold_trained_flag=0,
                                                                                                    what_labels_to_use=label, mid_layer_config=mid_layer, n_mid_neurons1=n_mid_neurons1,
                                                                                                    n_mid_neurons2=n_mid_neurons2))
                                                                        TL_MODEL_PARAMETERS.append(temp_model_dict.copy())
                                                                        n = n + 1

                                                                if 'tri' in which_model_mode_to_use:
                                                                    temp_model_dict.update(dict(ID=n, layer_config=layer, base_model=base_model, optimizer=optimizer, learning_rate=lr,
                                                                                                n_neurons1=n_neurons1, n_neurons2=n_neurons2, n_neurons3=n_neurons3, dropout=dropout,
                                                                                                freeze_base_model=freeze, base_model_pooling=base_model_pooling,
                                                                                                which_scale_to_use=['25x', '100x', '400x'], trained_epoches=0, early_stopping=0,
                                                                                                augment_multiplier=aug_mult, model_mode='tri', model_trained_flag=0, fold_trained_flag=0,
                                                                                                what_labels_to_use=label, mid_layer_config=mid_layer, n_mid_neurons1=n_mid_neurons1,
                                                                                                n_mid_neurons2=n_mid_neurons2))
                                                                    TL_MODEL_PARAMETERS.append(temp_model_dict.copy())
                                                                    n = n + 1

    return TL_MODEL_PARAMETERS, TL_MODELS_AND_LOSS_ARRAY


def get_mono_scale_model(img_width, img_height, n_channels, N_CLASSES, base_model, layer_config, n_neurons1,
                         n_neurons2, n_neurons3, freeze_base_model, base_model_pooling, dropout):
    # Model input
    image_input = Input(shape=(img_height, img_width, n_channels), name='input2')

    # Load base model
    if base_model == 'VGG16':
        # Check that input size is valid for this base model.
        if img_width >= 48 and img_height >= 48:
            TL_base_model = VGG16(input_tensor=image_input, include_top=False, weights='imagenet', pooling=None)
            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            last_layer = TL_base_model.get_layer('block5_pool').output
            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:
                flatten_layer = GlobalAveragePooling2D()(last_layer)
            elif base_model_pooling in ['MAX', 'Max', 'max']:
                flatten_layer = GlobalMaxPooling2D()(last_layer)
            elif base_model_pooling in ['NONE', 'None', 'none']:
                flatten_layer = Flatten(name='flatten')(last_layer)
            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 48. Stopping program.', error=True)
            exit()
    elif base_model == 'VGG19':
        # Check that input size is valid for this base model.
        if img_width >= 48 and img_height >= 48:
            TL_base_model = VGG19(input_tensor=image_input, include_top=False, weights='imagenet', pooling=None)
            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            last_layer = TL_base_model.get_layer('block5_pool').output
            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:
                flatten_layer = GlobalAveragePooling2D()(last_layer)
            elif base_model_pooling in ['MAX', 'Max', 'max']:
                flatten_layer = GlobalMaxPooling2D()(last_layer)
            elif base_model_pooling in ['NONE', 'None', 'none']:
                flatten_layer = Flatten(name='flatten')(last_layer)
            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 48. Stopping program.', error=True)
            exit()
    else:
        my_print('Error in transfer learning base_model. Please choose another base model. Stopping program.', error=True)
        exit()

    # Get size of "latent vector"
    latent_vector_size = 1
    if base_model_pooling in ['NONE', 'None', 'none']:
        for n in range(1, 4):
            latent_vector_size *= last_layer.get_shape().as_list()[n]
    elif base_model_pooling in ['AVG', 'Avg', 'avg', 'MAX', 'Max', 'max']:
        latent_vector_size = flatten_layer.get_shape().as_list()[1]

    # Freeze all convolutional layers
    if freeze_base_model is True or freeze_base_model == 'Hybrid':
        for layer in TL_base_model.layers:
            layer.trainable = False

    # Define new classifier architecture
    if layer_config == 'config0':
        my_dense = flatten_layer
    elif layer_config == 'config1':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
    elif layer_config == 'config2':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
    elif layer_config == 'config3':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dense(n_neurons3, activation='relu', name='my_dense3')(my_dense)
    elif layer_config == 'config1_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
    elif layer_config == 'config2_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
    elif layer_config == 'config3_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons3, activation='relu', name='my_dense3')(my_dense)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
    else:
        my_print('Error in layer_config. Please choose another layer.', error=True)
        exit()

    # OUTPUT CLASSIFIER LAYER
    my_output = Dense(N_CLASSES, activation='softmax', name='my_output')(my_dense)

    # Define the models
    TL_classifier_model = Model(image_input, my_output)

    return TL_classifier_model, latent_vector_size


def get_di_scale_model(img_width, img_height, n_channels, N_CLASSES, base_model, layer_config, n_neurons1,
                       n_neurons2, n_neurons3, freeze_base_model, base_model_pooling, dropout):
    # Model input
    image_input_400x = Input(shape=(img_width, img_height, n_channels), name='input_400x')
    image_input_100x = Input(shape=(img_width, img_height, n_channels), name='input_100x')

    if base_model == 'VGG16':
        # Check that input size is valid for this base model.
        if img_width >= 48 and img_height >= 48:

            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:

                base_model_400x = VGG16(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='avg')
                base_model_100x = VGG16(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='avg')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
            elif base_model_pooling in ['MAX', 'Max', 'max']:

                base_model_400x = VGG16(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='max')
                base_model_100x = VGG16(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='max')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
            elif base_model_pooling in ['NONE', 'None', 'none', None]:

                base_model_400x = VGG16(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling=None)
                base_model_100x = VGG16(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling=None)

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output

                # Get size of "latent vector"
                latent_vector_size = 1
                for n in range(1, 4):
                    latent_vector_size *= last_layer_400x.get_shape().as_list()[n]

                last_layer_400x = Flatten(name='flatten_400x')(last_layer_400x)
                last_layer_100x = Flatten(name='flatten_100x')(last_layer_100x)

            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 48. Stopping program.', error=True)
            exit()
    elif base_model == 'VGG19':
        # Check that input size is valid for this base model.
        if img_width >= 48 and img_height >= 48:

            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:

                base_model_400x = VGG19(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='avg')
                base_model_100x = VGG19(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='avg')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
            elif base_model_pooling in ['MAX', 'Max', 'max']:

                base_model_400x = VGG19(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='max')
                base_model_100x = VGG19(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='max')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
            elif base_model_pooling in ['NONE', 'None', 'none', None]:

                base_model_400x = VGG19(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling=None)
                base_model_100x = VGG19(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling=None)

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output

                # Get size of "latent vector"
                latent_vector_size = 1
                for n in range(1, 4):
                    latent_vector_size *= last_layer_400x.get_shape().as_list()[n]

                last_layer_400x = Flatten(name='flatten_400x')(last_layer_400x)
                last_layer_100x = Flatten(name='flatten_100x')(last_layer_100x)

            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 48. Stopping program.', error=True)
            exit()
    else:
        my_print('Error in transfer learning base_model. Please choose another base model. Stopping program.', error=True)
        exit()

    # Get size of "latent vector"
    if base_model_pooling in ['AVG', 'Avg', 'avg', 'MAX', 'Max', 'max']:
        latent_vector_size = last_layer_400x.get_shape().as_list()[1]

    # Rename all layers in first model
    for layer in base_model_100x.layers:
        layer.name = layer.name + str("_10x")

    # Freeze all convolutional layers
    if freeze_base_model is True or freeze_base_model == 'Hybrid':
        for layer in base_model_400x.layers:
            layer.trainable = False

        for layer in base_model_100x.layers:
            layer.trainable = False

    # Concatenate models
    flatten_layer = keras.layers.concatenate([last_layer_400x, last_layer_100x], axis=-1)

    # Define new classifier architecture
    if layer_config == 'config0':
        my_dense = flatten_layer
    elif layer_config == 'config1':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
    elif layer_config == 'config2':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
    elif layer_config == 'config3':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dense(n_neurons3, activation='relu', name='my_dense3')(my_dense)
    elif layer_config == 'config1_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
    elif layer_config == 'config2_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
    elif layer_config == 'config3_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons3, activation='relu', name='my_dense3')(my_dense)
    else:
        my_print('Error in layer_config. Please choose another layer.', error=True)
        exit()

    # OUTPUT CLASSIFIER LAYER
    my_output = Dense(N_CLASSES, activation='softmax', name='my_output')(my_dense)

    # Define the models
    TL_classifier_model = Model(inputs=[image_input_400x, image_input_100x], outputs=my_output)

    return TL_classifier_model, latent_vector_size


def get_tri_scale_model(img_width, img_height, n_channels, N_CLASSES, base_model, layer_config, n_neurons1,
                        n_neurons2, n_neurons3, freeze_base_model, base_model_pooling, dropout):
    # Model input
    image_input_400x = Input(shape=(img_width, img_height, n_channels), name='input_400x')
    image_input_100x = Input(shape=(img_width, img_height, n_channels), name='input_100x')
    image_input_25x = Input(shape=(img_width, img_height, n_channels), name='input_25x')

    if base_model == 'VGG16':
        # Check that input size is valid for this base model.
        if img_width >= 48 and img_height >= 48:

            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:

                base_model_400x = VGG16(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='avg')
                base_model_100x = VGG16(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='avg')
                base_model_25x = VGG16(input_tensor=image_input_25x, include_top=False, weights='imagenet', pooling='avg')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
                last_layer_25x = base_model_25x.layers[-1].output
            elif base_model_pooling in ['MAX', 'Max', 'max']:

                base_model_400x = VGG16(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='max')
                base_model_100x = VGG16(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='max')
                base_model_25x = VGG16(input_tensor=image_input_25x, include_top=False, weights='imagenet', pooling='max')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
                last_layer_25x = base_model_25x.layers[-1].output
            elif base_model_pooling in ['NONE', 'None', 'none', None]:

                base_model_400x = VGG16(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling=None)
                base_model_100x = VGG16(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling=None)
                base_model_25x = VGG16(input_tensor=image_input_25x, include_top=False, weights='imagenet', pooling=None)

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
                last_layer_25x = base_model_25x.layers[-1].output

                # Get size of "latent vector"
                latent_vector_size = 1
                for n in range(1, 4):
                    latent_vector_size *= last_layer_400x.get_shape().as_list()[n]

                last_layer_400x = Flatten(name='flatten_400x')(last_layer_400x)
                last_layer_100x = Flatten(name='flatten_100x')(last_layer_100x)
                last_layer_25x = Flatten(name='flatten_25x')(last_layer_25x)

            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 48. Stopping program.', error=True)
            exit()
    elif base_model == 'VGG19':
        # Check that input size is valid for this base model.
        if img_width >= 48 and img_height >= 48:

            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:

                base_model_400x = VGG19(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='avg')
                base_model_100x = VGG19(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='avg')
                base_model_25x = VGG19(input_tensor=image_input_25x, include_top=False, weights='imagenet', pooling='avg')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
                last_layer_25x = base_model_25x.layers[-1].output
            elif base_model_pooling in ['MAX', 'Max', 'max']:

                base_model_400x = VGG19(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='max')
                base_model_100x = VGG19(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='max')
                base_model_25x = VGG19(input_tensor=image_input_25x, include_top=False, weights='imagenet', pooling='max')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
                last_layer_25x = base_model_25x.layers[-1].output
            elif base_model_pooling in ['NONE', 'None', 'none', None]:

                base_model_400x = VGG19(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling=None)
                base_model_100x = VGG19(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling=None)
                base_model_25x = VGG19(input_tensor=image_input_25x, include_top=False, weights='imagenet', pooling=None)

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
                last_layer_25x = base_model_25x.layers[-1].output

                # Get size of "latent vector"
                latent_vector_size = 1
                for n in range(1, 4):
                    latent_vector_size *= last_layer_400x.get_shape().as_list()[n]

                last_layer_400x = Flatten(name='flatten_400x')(last_layer_400x)
                last_layer_100x = Flatten(name='flatten_100x')(last_layer_100x)
                last_layer_25x = Flatten(name='flatten_25x')(last_layer_25x)

            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 48. Stopping program.', error=True)
            exit()
    else:
        my_print('Error in transfer learning base_model. Please choose another base model. Stopping program.', error=True)
        exit()

    # Get size of "latent vector"
    if base_model_pooling in ['AVG', 'Avg', 'avg', 'MAX', 'Max', 'max']:
        latent_vector_size = last_layer_400x.get_shape().as_list()[1]

    # Rename all layers in first model
    for layer in base_model_100x.layers:
        layer.name = layer.name + str("_100x")

    # Rename all layers in second model
    for layer in base_model_25x.layers:
        layer.name = layer.name + str("_25x")

    # Freeze all convolutional layers
    if freeze_base_model is True or freeze_base_model == 'Hybrid':
        for layer in base_model_400x.layers:
            layer.trainable = False

        for layer in base_model_100x.layers:
            layer.trainable = False

        for layer in base_model_25x.layers:
            layer.trainable = False

    # Concatenate models
    flatten_layer = keras.layers.concatenate([last_layer_400x, last_layer_100x, last_layer_25x], axis=-1)

    # Define new classifier architecture
    if layer_config == 'config0':
        my_dense = flatten_layer
    elif layer_config == 'config1':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
    elif layer_config == 'config2':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
    elif layer_config == 'config3':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dense(n_neurons3, activation='relu', name='my_dense3')(my_dense)
    elif layer_config == 'config1_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
    elif layer_config == 'config2_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
    elif layer_config == 'config3_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons3, activation='relu', name='my_dense3')(my_dense)
    else:
        my_print('Error in layer_config. Please choose another layer.', error=True)
        exit()

    # OUTPUT CLASSIFIER LAYER
    my_output = Dense(N_CLASSES, activation='softmax', name='my_output')(my_dense)

    # Define the models
    TL_classifier_model = Model(inputs=[image_input_400x, image_input_100x, image_input_25x], outputs=my_output)

    return TL_classifier_model, latent_vector_size


def plot_confusion_matrix(cm, epoch, classes, SUMMARY_PATH, folder_name, title='Confusion matrix', cmap=plt.cm.Blues):
    # This function prints and plots the confusion matrix.
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # plot_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
    plot_filepath = '{}{}/'.format(SUMMARY_PATH, folder_name)
    os.makedirs(plot_filepath, exist_ok=True)
    plt.savefig(plot_filepath + 'Confusion_matrix_epoch_' + str(epoch) + '.png', dpi=200)
    plt.close()
    plt.cla()


def save_history_plot(history, path, mode, model_no):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    title = '{} - model {} - accuracy'.format(mode, model_no)
    plt.title(title)
    plt.grid()
    plt.ylabel('accuracy')
    # plt.ylim(0.6, 1.05)  # set the ylim to ymin, ymax
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plot_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
    plt.savefig(path + 'accuracy_plot_ ' + plot_timestamp + '.png', dpi=200)
    plt.close()
    plt.cla()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    title = '{} - model {} - loss'.format(mode, model_no)
    plt.title(title)
    plt.grid()
    plt.ylabel('loss')
    # plt.ylim(-0.5, 1)  # set the ylim to ymin, ymax
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(path + 'loss_plot_' + plot_timestamp + '.png', dpi=200)
    plt.close()
    plt.cla()


def save_learning_rate_history_plot(history, path, mode, model_no):
    # summarize history for loss
    plt.plot(history)
    title = '{} - model {} - loss'.format(mode, model_no)
    plt.title(title)
    plt.grid()
    plt.ylabel('Learning rate')
    # plt.ylim(-0.5, 1)  # set the ylim to ymin, ymax
    plt.xlabel('epoch')
    plot_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
    plt.savefig(path + 'lr_plot_' + plot_timestamp + '.png', dpi=200)
    plt.close()
    plt.cla()


class mode_8_mono_coordinates_generator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, tile_dicts, batch_size, n_classes, shuffle, TILE_SIZE, which_scale_to_use, label, base_model,
                 PRE_DIVIDE_TILES_BY_255, PRE_RGB_TO_BGR, PRE_SUBTRACT_MEAN_FROM_TILE):
        """Initialization.

        Args:
            img_files: A list of path to image files.
            labels: A dictionary of corresponding labels.
        """
        self.tile_dicts = tile_dicts
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.TILE_SIZE = TILE_SIZE
        self.label = label
        self.on_epoch_end()
        self.base_model = base_model

        self.PRE_DIVIDE_TILES_BY_255 = PRE_DIVIDE_TILES_BY_255
        self.PRE_RGB_TO_BGR = PRE_RGB_TO_BGR
        self.PRE_SUBTRACT_MEAN_FROM_TILE = PRE_SUBTRACT_MEAN_FROM_TILE

        if which_scale_to_use == '400x':
            self.img_one_level = 0
            self.coordinate_scale = 'coordinates_400x'
        elif which_scale_to_use == '100x':
            self.img_one_level = 1
            self.coordinate_scale = 'coordinates_100x'
        elif which_scale_to_use == '25x':
            self.img_one_level = 2
            self.coordinate_scale = 'coordinates_25x'

    def __len__(self):
        """Denotes the number of batches per epoch."""
        # Added one to allow last batch smaller
        # return int(np.floor(len(self.tile_dicts) / self.batch_size)) + 1
        return int(np.ceil(len(self.tile_dicts) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch

        # Modified to allow smaller last batch
        if ((index + 1) * self.batch_size) > len(self.indexes):
            indexes = self.indexes[index * self.batch_size:]
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        tile_dicts_temp = [self.tile_dicts[k] for k in indexes]

        # Generate data
        X_400x, y = self.__data_generation(tile_dicts_temp)

        return X_400x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.tile_dicts))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, tile_dicts_temp):
        """Generates data containing batch_size samples."""
        X_img = []
        y = np.empty((len(tile_dicts_temp)), dtype=int)

        # Generate data
        for i, tile_dict in enumerate(tile_dicts_temp):
            # Read image, flatten and rotate 90 degree so that the coordinated line up correctly
            full_image = pyvips.Image.new_from_file(tile_dict['path'], level=self.img_one_level).flatten().rot(1)

            # Extract tile
            tile = full_image.extract_area(tile_dict[self.coordinate_scale][0], tile_dict[self.coordinate_scale][1], self.TILE_SIZE, self.TILE_SIZE)

            # Augmentation
            tile = augment_tiles(aug_argument=tile_dict['augmentarg'], tile=tile)

            # Preprocess. Normalize from 0-255 to 0-1.
            if self.PRE_DIVIDE_TILES_BY_255:
                tile = dataset_preprocessing(preprocess_mode=1, input_tile=tile)

            # Write tile to memory and convert to numpy array
            tile_numpy = np.ndarray(buffer=tile.write_to_memory(),
                                    dtype=my_constants.format_to_dtype[tile.format],
                                    shape=[tile.height, tile.width, tile.bands])

            # Preprocess. Subtract mean
            if self.PRE_SUBTRACT_MEAN_FROM_TILE:
                tile_numpy = dataset_preprocessing(preprocess_mode=3, input_tile=tile_numpy)

            # Preprocess. 'RGB'->'BGR'.
            if self.PRE_RGB_TO_BGR:
                tile_numpy = dataset_preprocessing(preprocess_mode=2, input_tile=tile_numpy)

            X_img.append(tile_numpy)
            y[i] = tile_dict[self.label]

        X_tile_array = np.array(X_img)
        return X_tile_array, keras.utils.to_categorical(y, num_classes=self.n_classes)


class mode_8_di_coordinates_generator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, tile_dicts, batch_size, n_classes, shuffle, TILE_SIZE, which_scale_to_use, label,
                 PRE_DIVIDE_TILES_BY_255, PRE_RGB_TO_BGR, PRE_SUBTRACT_MEAN_FROM_TILE):
        """Initialization.

        Args:
            img_files: A list of path to image files.
            labels: A dictionary of corresponding labels.
        """
        self.tile_dicts = tile_dicts
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.tile_size = TILE_SIZE
        self.which_scale_to_use = which_scale_to_use
        self.label = label
        self.on_epoch_end()

        self.PRE_DIVIDE_TILES_BY_255 = PRE_DIVIDE_TILES_BY_255
        self.PRE_RGB_TO_BGR = PRE_RGB_TO_BGR
        self.PRE_SUBTRACT_MEAN_FROM_TILE = PRE_SUBTRACT_MEAN_FROM_TILE

        if which_scale_to_use[0] == '400x':
            self.img_one_level = 0
            self.coordinate_one_scale = 'coordinates_400x'
        elif which_scale_to_use[0] == '100x':
            self.img_one_level = 1
            self.coordinate_one_scale = 'coordinates_100x'
        elif which_scale_to_use[0] == '25x':
            self.img_one_level = 2
            self.coordinate_one_scale = 'coordinates_25x'

        if which_scale_to_use[1] == '400x':
            self.img_two_level = 0
            self.coordinate_two_scale = 'coordinates_400x'
        elif which_scale_to_use[1] == '100x':
            self.img_two_level = 1
            self.coordinate_two_scale = 'coordinates_100x'
        elif which_scale_to_use[1] == '25x':
            self.img_two_level = 2
            self.coordinate_two_scale = 'coordinates_25x'

    def __len__(self):
        """Denotes the number of batches per epoch."""
        # Added one to allow last batch smaller
        # return int(np.floor(len(self.tile_dicts) / self.batch_size)) + 1
        return int(np.ceil(len(self.tile_dicts) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch

        # Modified to allow smaller last batch
        if ((index + 1) * self.batch_size) > len(self.indexes):
            indexes = self.indexes[index * self.batch_size:]
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        tile_dicts_temp = [self.tile_dicts[k] for k in indexes]

        # Generate data
        [X_one, X_two], y = self.__data_generation(tile_dicts_temp)

        return [X_one, X_two], y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.tile_dicts))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, tile_dicts_temp):
        """Generates data containing batch_size samples."""
        X_img_one = []
        X_img_two = []
        y = np.empty((len(tile_dicts_temp)), dtype=int)

        # Generate data
        for i, tile_dict in enumerate(tile_dicts_temp):
            # Read image, flatten and rotate image 90 degree so that the coordinated line up correctly
            full_image_one = pyvips.Image.new_from_file(tile_dict['path'], level=self.img_one_level).flatten().rot(1)
            full_image_two = pyvips.Image.new_from_file(tile_dict['path'], level=self.img_two_level).flatten().rot(1)

            # Extract tile
            tile_one = full_image_one.extract_area(tile_dict[self.coordinate_one_scale][0], tile_dict[self.coordinate_one_scale][1], self.tile_size, self.tile_size)
            tile_two = full_image_two.extract_area(tile_dict[self.coordinate_two_scale][0], tile_dict[self.coordinate_two_scale][1], self.tile_size, self.tile_size)

            # Augmentation
            tile_one = augment_tiles(aug_argument=tile_dict['augmentarg'], tile=tile_one)
            tile_two = augment_tiles(aug_argument=tile_dict['augmentarg'], tile=tile_two)

            # Preprocess. Normalize from 0-255 to 0-1.
            if self.PRE_DIVIDE_TILES_BY_255:
                tile_one = dataset_preprocessing(preprocess_mode=1, input_tile=tile_one)
                tile_two = dataset_preprocessing(preprocess_mode=1, input_tile=tile_two)

            # Write tile to memory and convert to numpy array
            tile_one_numpy = np.ndarray(buffer=tile_one.write_to_memory(),
                                        dtype=my_constants.format_to_dtype[tile_one.format],
                                        shape=[tile_one.height, tile_one.width, tile_one.bands])

            # Write tile to memory and convert to numpy array
            tile_two_numpy = np.ndarray(buffer=tile_two.write_to_memory(),
                                        dtype=my_constants.format_to_dtype[tile_two.format],
                                        shape=[tile_two.height, tile_two.width, tile_two.bands])

            # Preprocess. Subtract mean
            if self.PRE_SUBTRACT_MEAN_FROM_TILE:
                tile_one_numpy = dataset_preprocessing(preprocess_mode=3, input_tile=tile_one_numpy)
                tile_two_numpy = dataset_preprocessing(preprocess_mode=3, input_tile=tile_two_numpy)

            # Preprocess. 'RGB'->'BGR'.
            if self.PRE_RGB_TO_BGR:
                tile_one_numpy = dataset_preprocessing(preprocess_mode=2, input_tile=tile_one_numpy)
                tile_two_numpy = dataset_preprocessing(preprocess_mode=2, input_tile=tile_two_numpy)

            X_img_one.append(tile_one_numpy)
            X_img_two.append(tile_two_numpy)
            y[i] = tile_dict[self.label]

        X_tile_one_array = np.array(X_img_one)
        X_tile_two_array = np.array(X_img_two)
        return [X_tile_one_array, X_tile_two_array], keras.utils.to_categorical(y, num_classes=self.n_classes)


class mode_8_tri_coordinates_generator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, tile_dicts, batch_size, n_classes, shuffle, TILE_SIZE, which_scale_to_use, label,
                 PRE_DIVIDE_TILES_BY_255, PRE_RGB_TO_BGR, PRE_SUBTRACT_MEAN_FROM_TILE):
        """Initialization.

        Args:
            img_files: A list of path to image files.
            labels: A dictionary of corresponding labels.
        """
        self.tile_dicts = tile_dicts
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.tile_size = TILE_SIZE
        self.which_scale_to_use = which_scale_to_use
        self.label = label
        self.on_epoch_end()
        self.PRE_DIVIDE_TILES_BY_255 = PRE_DIVIDE_TILES_BY_255
        self.PRE_RGB_TO_BGR = PRE_RGB_TO_BGR
        self.PRE_SUBTRACT_MEAN_FROM_TILE = PRE_SUBTRACT_MEAN_FROM_TILE

    def __len__(self):
        """Denotes the number of batches per epoch."""
        # Added one to allow last batch smaller
        # return int(np.floor(len(self.tile_dicts) / self.batch_size)) + 1
        return int(np.ceil(len(self.tile_dicts) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch

        # Modified to allow smaller last batch
        if ((index + 1) * self.batch_size) > len(self.indexes):
            indexes = self.indexes[index * self.batch_size:]
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        tile_dicts_temp = [self.tile_dicts[k] for k in indexes]

        # Generate data
        [X_one, X_two, X_three], y = self.__data_generation(tile_dicts_temp)

        return [X_one, X_two, X_three], y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.tile_dicts))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, tile_dicts_temp):
        """Generates data containing batch_size samples."""
        X_img_400x = []
        X_img_100x = []
        X_img_25x = []
        y = np.empty((len(tile_dicts_temp)), dtype=int)

        # Generate data
        for i, tile_dict in enumerate(tile_dicts_temp):
            # Read image, flatten and rotate image 90 degree so that the coordinated line up correctly
            full_image_400x = pyvips.Image.new_from_file(tile_dict['path'], level=0).flatten().rot(1)
            full_image_100x = pyvips.Image.new_from_file(tile_dict['path'], level=1).flatten().rot(1)
            full_image_25x = pyvips.Image.new_from_file(tile_dict['path'], level=2).flatten().rot(1)

            # Extract tile
            tile_400x = full_image_400x.extract_area(tile_dict['coordinates_400x'][0], tile_dict['coordinates_400x'][1], self.tile_size, self.tile_size)
            tile_100x = full_image_100x.extract_area(tile_dict['coordinates_100x'][0], tile_dict['coordinates_100x'][1], self.tile_size, self.tile_size)
            tile_25x = full_image_25x.extract_area(tile_dict['coordinates_25x'][0], tile_dict['coordinates_25x'][1], self.tile_size, self.tile_size)

            # Augmentation
            tile_400x = augment_tiles(aug_argument=tile_dict['augmentarg'], tile=tile_400x)
            tile_100x = augment_tiles(aug_argument=tile_dict['augmentarg'], tile=tile_100x)
            tile_25x = augment_tiles(aug_argument=tile_dict['augmentarg'], tile=tile_25x)

            # Preprocess. Normalize from 0-255 to 0-1.
            if self.PRE_DIVIDE_TILES_BY_255:
                tile_400x = dataset_preprocessing(preprocess_mode=1, input_tile=tile_400x)
                tile_100x = dataset_preprocessing(preprocess_mode=1, input_tile=tile_100x)
                tile_25x = dataset_preprocessing(preprocess_mode=1, input_tile=tile_25x)

            # Write tile to memory and convert to numpy array
            tile_400x_numpy = np.ndarray(buffer=tile_400x.write_to_memory(),
                                         dtype=my_constants.format_to_dtype[tile_400x.format],
                                         shape=[tile_400x.height, tile_400x.width, tile_400x.bands])

            tile_100x_numpy = np.ndarray(buffer=tile_100x.write_to_memory(),
                                         dtype=my_constants.format_to_dtype[tile_100x.format],
                                         shape=[tile_100x.height, tile_100x.width, tile_100x.bands])

            tile_25x_numpy = np.ndarray(buffer=tile_25x.write_to_memory(),
                                        dtype=my_constants.format_to_dtype[tile_25x.format],
                                        shape=[tile_25x.height, tile_25x.width, tile_25x.bands])

            # Preprocess. Subtract mean
            if self.PRE_SUBTRACT_MEAN_FROM_TILE:
                tile_400x_numpy = dataset_preprocessing(preprocess_mode=3, input_tile=tile_400x_numpy)
                tile_100x_numpy = dataset_preprocessing(preprocess_mode=3, input_tile=tile_100x_numpy)
                tile_25x_numpy = dataset_preprocessing(preprocess_mode=3, input_tile=tile_25x_numpy)

            # Preprocess. 'RGB'->'BGR'.
            if self.PRE_RGB_TO_BGR:
                tile_400x_numpy = dataset_preprocessing(preprocess_mode=2, input_tile=tile_400x_numpy)
                tile_100x_numpy = dataset_preprocessing(preprocess_mode=2, input_tile=tile_100x_numpy)
                tile_25x_numpy = dataset_preprocessing(preprocess_mode=2, input_tile=tile_25x_numpy)

            X_img_400x.append(tile_400x_numpy)
            X_img_100x.append(tile_100x_numpy)
            X_img_25x.append(tile_25x_numpy)
            y[i] = tile_dict[self.label]

        X_tile_400x_array = np.array(X_img_400x)
        X_tile_100x_array = np.array(X_img_100x)
        X_tile_25x_array = np.array(X_img_25x)
        return [X_tile_400x_array, X_tile_100x_array, X_tile_25x_array], keras.utils.to_categorical(y, num_classes=self.n_classes)
