from sklearn.metrics import classification_report
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
from keras.optimizers import Adadelta
from keras.callbacks import Callback
from keras.optimizers import Adagrad
from keras.optimizers import Adamax
from keras.utils import plot_model
from keras.optimizers import Nadam
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import backend as K
import my_functions
import my_constants
import numpy as np
import pickle
import random
import keras
import glob
import sys
import csv
import time
import os


def transfer_learning_model(OPTIMIZER, batch_size, current_run_path, METADATA_FOLDER, MODEL_WEIGHT_FOLDER, N_CHANNELS,
                            N_WORKERS, EPOCHES, CLASSIFICATION_REPORT_FOLDER, layer_config, freeze_base_model, learning_rate, n_neurons_first_layer,
                            n_neurons_second_layer, n_neurons_third_layer, dropout, base_model, MODE_FOLDER, START_NEW_MODEL, EARLY_STOPPING_PATIENCE,
                            WHAT_LABELS_TO_USE, base_model_pooling, ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE,
                            MODELS_AND_LOSS_ARRAY_DTL_PICKLE_FILE, TRAINING_DATA_DTL_PICKLE_FILE, REDUCE_LR_PATIENCE, ACTIVATE_ReduceLROnPlateau,
                            USE_MULTIPROCESSING, MAX_QUEUE_SIZE, SMALL_DATASET_DEBUG_MODE, DIAGNOSTIC_TRAINING_DICTS_PATH, which_model_mode_to_use,
                            which_scale_to_use_mono, which_scale_to_use_di, SCN_PATH, SUMMARY_CSV_FILE_PATH, tf_version, TILE_SIZE_DIAGNOSTIC,
                            blocks_to_unfreeze_vgg16_vgg19, delayed_unfreeze_start_epoche_vgg16_vgg19, DIAGNOSTIC_VALIDATION_DICTS_PATH,
                            SAVE_CONF_MAT_AND_REPORT_ON_END, PRE_DIVIDE_TILES_BY_255, PRE_RGB_TO_BGR, PRE_SUBTRACT_MEAN_FROM_TILE,
                            BALANCE_DATASET_BY_AUGMENTATION, AUGMENT_MINORITY_WSI, DESIRED_TILES_IN_EACH_WSI):
    # region FILE INIT

    # Check that variable contain valid values
    for n in freeze_base_model:
        assert n in [True, False, 'Hybrid'], 'Variable freeze_base_model must be one of: [True, False, \'Hybrid\']. Now it is: {}'.format(n)

    for n in WHAT_LABELS_TO_USE:
        assert n in ['Recurrence', 'Progression', 'Stage', 'WHO04', 'WHO73'], 'Variable WHAT_LABELS_TO_USE must be one of: [\'Recurrence\', \'Progression\', \'Stage\', \'WHO04\', \'WHO73\']. Now it is: {}'.format(n)

    for n in blocks_to_unfreeze_vgg16_vgg19:
        assert n in ['block1', 'block2', 'block3', 'block4', 'block5'], 'Variable blocks_to_unfreeze_vgg16_vgg19 must be one of: [\'block1\', \'block2\', \'block3\', \'block4\', \'block5\']. Now it is: {}'.format(n)

    for n in which_model_mode_to_use:
        assert n in ['mono', 'di', 'tri'], 'Variable which_model_mode_to_use must be one of: [\'mono\', \'di\', \'tri\']. Now it is: {}'.format(n)

    for n in which_scale_to_use_mono:
        assert n in ['25x', '100x', '400x'], 'Variable which_scale_to_use_mono must be one of: [\'25x\', \'100x\', \'400x\']. Now it is: {}'.format(n)

    for n in which_scale_to_use_di:
        assert n in [['100x', '400x'], ['25x', '400x'], ['25x', '100x']], 'Variable which_scale_to_use_di must be one of: [[\'100x\', \'400x\'], [\'25x\', \'400x\'], [\'25x\', \'100x\']]. Now it is: {}'.format(n)

    # Check if we are using learning rate decay on plateau (for summary csv file)
    if ACTIVATE_ReduceLROnPlateau:
        ReduceLRstatus = str(REDUCE_LR_PATIENCE)
    else:
        ReduceLRstatus = 'False'
    # endregion

    # region CREATE OR LOAD LIST OF MODELS
    # Check if we have previously generated list of classifier models (if we continue training existing models)
    if os.path.isfile(current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE):
        # File exist, load parameters.
        ALL_MODEL_PARAMETERS = my_functions.pickle_load(current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE)
        MODELS_AND_LOSS_ARRAY = my_functions.pickle_load(current_run_path + METADATA_FOLDER + MODELS_AND_LOSS_ARRAY_DTL_PICKLE_FILE)
    else:
        # File does not exist, generate new lists of models
        ALL_MODEL_PARAMETERS, MODELS_AND_LOSS_ARRAY = my_functions.list_of_CVTL_models(LAYER_CONFIG=layer_config,
                                                                                       BASE_MODEL=base_model,
                                                                                       OPTIMIZER=OPTIMIZER,
                                                                                       LEARNING_RATE=learning_rate,
                                                                                       N_NEURONS_FIRST_LAYER=n_neurons_first_layer,
                                                                                       N_NEURONS_SECOND_LAYER=n_neurons_second_layer,
                                                                                       N_NEURONS_THIRD_LAYER=n_neurons_third_layer,
                                                                                       DROPOUT=dropout,
                                                                                       freeze_base_model=freeze_base_model,
                                                                                       BASE_MODEL_POOLING=base_model_pooling,
                                                                                       which_model_mode_to_use=which_model_mode_to_use,
                                                                                       which_scale_to_use_mono=which_scale_to_use_mono,
                                                                                       which_scale_to_use_di=which_scale_to_use_di,
                                                                                       augmentation_multiplier=[1],
                                                                                       WHAT_LABELS_TO_USE=WHAT_LABELS_TO_USE,
                                                                                       middle_layer_config=[0],
                                                                                       n_neurons_mid_first_layer=[0],
                                                                                       n_neurons_mid_second_layer=[0])

        # Save to file
        my_functions.pickle_save(ALL_MODEL_PARAMETERS, current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE)
        my_functions.pickle_save(MODELS_AND_LOSS_ARRAY, current_run_path + METADATA_FOLDER + MODELS_AND_LOSS_ARRAY_DTL_PICKLE_FILE)
    # endregion

    # Loop through all models to train
    for current_model_dict in ALL_MODEL_PARAMETERS:

        # Check if model already have been trained. If number of trained_epoches is the same as number of epoches, then the model have been trained.
        if (current_model_dict['trained_epoches'] < EPOCHES) and (current_model_dict['early_stopping'] is 0):

            # region MODEL INIT

            # Read hyperparameters of current model
            current_model_no = current_model_dict['ID']
            current_base_model = current_model_dict['base_model']
            current_layer_config = current_model_dict['layer_config']
            current_optimizer = current_model_dict['optimizer']
            current_learning_rate = current_model_dict['learning_rate']
            current_n_neurons1 = current_model_dict['n_neurons1']
            current_n_neurons2 = current_model_dict['n_neurons2']
            current_n_neurons3 = current_model_dict['n_neurons3']
            current_dropout = current_model_dict['dropout']
            current_freeze_base_model = current_model_dict['freeze_base_model']
            current_base_model_pooling = current_model_dict['base_model_pooling']
            current_scale_to_use = current_model_dict['which_scale_to_use']
            current_model_mode = current_model_dict['model_mode']
            current_label_to_use = current_model_dict['what_labels_to_use']

            # Set correct name and number of classes
            name_and_index_of_classes = my_constants.get_diagnostic_name_and_index_of_classes(label_to_use=current_label_to_use)
            # N_CLASSES_ALL = len(name_and_index_of_classes)
            N_CLASSES_TRAINING = sum(1 for _, tile in name_and_index_of_classes.items() if tile['used_in_training'] == 1)
            # index_of_background = [index for index, tile in name_and_index_of_classes.items() if tile['name'] == 'background'][0]
            # index_of_undefined = [index for index, tile in name_and_index_of_classes.items() if tile['name'] == 'undefined'][0]

            NAME_OF_CLASSES_ALL = []
            NAME_OF_CLASSES_ALL_DISPLAYNAME = []
            NAME_OF_CLASSES_TRAINING = []
            NAME_OF_CLASSES_TRAINING_DISPLAYNAME = []
            for index, value in name_and_index_of_classes.items():
                NAME_OF_CLASSES_ALL.append(value['name'])
                NAME_OF_CLASSES_ALL_DISPLAYNAME.append(value['display_name'])
                if value['used_in_training'] == 1:
                    NAME_OF_CLASSES_TRAINING.append(value['name'])
                    NAME_OF_CLASSES_TRAINING_DISPLAYNAME.append(value['display_name'])

            # Create list of labels for confusion matrix
            cm_label = list(range(N_CLASSES_TRAINING))

            # Make a vector with name of each class
            name_of_classes_array = []
            for index, class_name in enumerate(NAME_OF_CLASSES_TRAINING):
                name_of_classes_array.append('Class ' + str(index) + ': ' + class_name)

            # Define name of the model
            if current_model_mode == 'mono':
                current_model_name = 'Diagnostic_Model_' + str(current_model_no) + '_' + current_model_mode + '_' + current_scale_to_use + '/'
            elif str(current_model_mode) == 'di':
                current_model_name = 'Diagnostic_Model_' + str(current_model_no) + '_' + current_model_mode + '_' + current_scale_to_use[0] + '_' + current_scale_to_use[1] + '/'
            elif current_model_mode == 'tri':
                current_model_name = 'Diagnostic_Model_' + str(current_model_no) + '_' + current_model_mode + '_25x_100x_400x/'

            # Define summary path
            current_mode_summary_path = current_run_path + MODE_FOLDER
            os.makedirs(current_mode_summary_path, exist_ok=True)

            # Make folder for new model.
            current_model_summary_path = current_mode_summary_path + current_model_name
            os.makedirs(current_model_summary_path, exist_ok=True)

            # Make a classification report folder
            if SAVE_CONF_MAT_AND_REPORT_ON_END is True:
                current_model_classification_report_path = current_model_summary_path + CLASSIFICATION_REPORT_FOLDER
                os.makedirs(current_model_classification_report_path, exist_ok=True)

            # Create folder to save models weight
            weight_save_path = current_run_path + MODEL_WEIGHT_FOLDER + current_model_name
            os.makedirs(weight_save_path, exist_ok=True)

            # Create a log to save epoches/accuracy/loss
            csv_logger = my_functions.create_keras_logger(current_mode_summary_path, current_model_name)

            # Make an array for top row in summary CSV file containing IMAGE_NAME and name of each class
            TILES_IN_WSI_SUMMARY_CSV_FILE = 'tiles_in_wsi_summary_csv_file.csv'
            csv_array = ['Image_Name', 'n_tiles_in_current_wsi_before_aug', 'n_tiles_in_current_wsi_after_aug',
                         'DESIRED_TILES_IN_EACH_WSI', 'current_augment_multiplier', 'temp_aug_multiplier']

            # Create a new csv file
            if not os.path.isfile(current_mode_summary_path + TILES_IN_WSI_SUMMARY_CSV_FILE):
                try:
                    with open(current_mode_summary_path + TILES_IN_WSI_SUMMARY_CSV_FILE, 'w') as csvfile:
                        csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(csv_array)
                except Exception as e:
                    my_functions.my_print('Error writing to file', error=True)
                    my_functions.my_print(e, error=True)

            my_functions.my_print('')
            my_functions.my_print('DTL-model {} of {}'.format(current_model_dict['ID'], len(ALL_MODEL_PARAMETERS) - 1))
            my_functions.my_print('\tModel_mode:{}, Scale:{}'.format(current_model_mode, current_scale_to_use))
            my_functions.my_print('\tBase_model:{}, freeze:{}, Model_pooling:{}, Layer_config:{}, optimizer:{}'.format(current_base_model, current_freeze_base_model,
                                                                                                                       current_base_model_pooling, current_layer_config,
                                                                                                                       current_optimizer))
            my_functions.my_print('\tlearning rate:{}, batch size:{}, n_neurons1:{}, n_neurons2:{}, n_neurons3:{}, dropout:{}'.format(current_learning_rate, batch_size,
                                                                                                                                      current_n_neurons1, current_n_neurons2,
                                                                                                                                      current_n_neurons3, current_dropout))
            # endregion

            # region DATASET INIT
            # List path of all files in a list
            wsi_list_training = os.listdir(DIAGNOSTIC_TRAINING_DICTS_PATH)
            wsi_list_training.sort()
            wsi_list_validation = os.listdir(DIAGNOSTIC_VALIDATION_DICTS_PATH)
            wsi_list_validation.sort()

            # Create a new dict to rule them all
            training_dict = dict()
            validation_dict = dict()
            wsi_class_counter_training = []
            wsi_class_counter_validation = []
            temp_aug_multiplier = -1
            index_to_tiles_to_augment = []
            index_to_tiles_to_augment_extra = []
            current_augment_multiplier = -1

            # Get training data. Go through each patient and extract tiles
            for current_wsi in wsi_list_training:
                # Restore coordinate data
                list_of_region_dicts_in_current_wsi = my_functions.pickle_load(DIAGNOSTIC_TRAINING_DICTS_PATH + current_wsi)
                """
                Example of list_of_region_dicts_in_current_wsi:
                    [{  'tissue_type': 'urothelium', 
                        'wsi_filename': 'H3374-10A HES_2015-08-20 10_12_19.scn', 
                        'WHO73': 0, 
                        'Recurrence': 1, 
                        'Stage': 0, 
                        'Progression': 0, 
                        'WHO04': 1,  
                        'tile_dict': {0: {'coordinates_400x': (162399, 47262), 'coordinates_100x': (40502, 11718), 'coordinates_25x': (10029, 2831)}, 1:{...}, 2: ... } }]
                """

                # Add data to main dict
                wsi_class_counter_training.append(list_of_region_dicts_in_current_wsi[0][current_label_to_use])

                # Count number of tiles in current WSI (to check if we need augmentation)
                n_tiles_in_current_wsi_before_aug = 0
                n_tiles_in_current_wsi_after_aug = 0
                for current_region in list_of_region_dicts_in_current_wsi:
                    n_tiles_in_current_wsi_before_aug += len(current_region['tile_dict'].items())

                # Augment tiles in WSI with less than DESIRED_TILES_IN_EACH_WSI tiles
                if AUGMENT_MINORITY_WSI:
                    # Calculate how many tiles needs to be augmented to get closer to desired number of tiles
                    current_augment_multiplier = DESIRED_TILES_IN_EACH_WSI // n_tiles_in_current_wsi_before_aug

                    if current_augment_multiplier == 1:
                        n_tiles_to_augment = DESIRED_TILES_IN_EACH_WSI - n_tiles_in_current_wsi_before_aug
                        index_to_tiles_to_augment = list(range(n_tiles_in_current_wsi_before_aug))
                        index_to_tiles_to_augment = random.sample(list(index_to_tiles_to_augment), n_tiles_to_augment)
                        index_to_tiles_to_augment_extra = []
                    elif current_augment_multiplier in [2, 3, 4, 5, 6, 7]:
                        index_to_tiles_to_augment = list(range(n_tiles_in_current_wsi_before_aug))
                        n_tiles_to_augment = DESIRED_TILES_IN_EACH_WSI - (n_tiles_in_current_wsi_before_aug * current_augment_multiplier)
                        index_to_tiles_to_augment_extra = list(range(n_tiles_in_current_wsi_before_aug))
                        index_to_tiles_to_augment_extra = random.sample(list(index_to_tiles_to_augment_extra), n_tiles_to_augment)
                    elif current_augment_multiplier >= 8:
                        # Maximum augmentation we can do is 8
                        current_augment_multiplier = 8
                        index_to_tiles_to_augment = list(range(n_tiles_in_current_wsi_before_aug))
                        index_to_tiles_to_augment_extra = []
                    elif current_augment_multiplier < 1:
                        current_augment_multiplier = 0
                        index_to_tiles_to_augment = []
                        index_to_tiles_to_augment_extra = []

                # Add tiles to dataset
                for current_region in list_of_region_dicts_in_current_wsi:

                    for tile_index, value in current_region['tile_dict'].items():
                        """
                        Example of value:
                            'value': {
                            0: {
                                'coordinates_400x': (162399, 47262), 
                                'coordinates_100x': (40502, 11718), 
                                'coordinates_25x': (10029, 2831)
                            }, 
                            1:{...}, 2: ... } }
                        """

                        # Check if class belongs to one of the classes to augment. If yes, add n copies to dict.
                        if tile_index in index_to_tiles_to_augment:

                            if (tile_index in index_to_tiles_to_augment_extra) or (current_augment_multiplier == 1):
                                temp_aug_multiplier = current_augment_multiplier + 1
                            else:
                                temp_aug_multiplier = current_augment_multiplier

                            for n in range(temp_aug_multiplier):
                                # Augment
                                tempvalue = value.copy()
                                tempvalue['augmentarg'] = n
                                tempvalue['path'] = SCN_PATH + current_region['wsi_filename'].split(".")[0] + '/' + current_region['wsi_filename']
                                tempvalue['WHO73'] = current_region['WHO73']
                                tempvalue['Recurrence'] = current_region['Recurrence']
                                tempvalue['Stage'] = current_region['Stage']
                                tempvalue['Progression'] = current_region['Progression']
                                tempvalue['WHO04'] = current_region['WHO04']
                                training_dict[len(training_dict)] = tempvalue
                                n_tiles_in_current_wsi_after_aug += 1
                        else:
                            # No augmentation, add to dict.
                            tempvalue = value.copy()
                            tempvalue['augmentarg'] = 1
                            tempvalue['path'] = SCN_PATH + current_region['wsi_filename'].split(".")[0] + '/' + current_region['wsi_filename']
                            tempvalue['WHO73'] = current_region['WHO73']
                            tempvalue['Recurrence'] = current_region['Recurrence']
                            tempvalue['Stage'] = current_region['Stage']
                            tempvalue['Progression'] = current_region['Progression']
                            tempvalue['WHO04'] = current_region['WHO04']
                            training_dict[len(training_dict)] = tempvalue
                            n_tiles_in_current_wsi_after_aug += 1

                        # break out from regions (for debugging)
                        if SMALL_DATASET_DEBUG_MODE is True and tile_index >= 512:
                            break

                    # break out from regions (for debugging)
                    #if SMALL_DATASET_DEBUG_MODE is True and len(training_dict) >= 512:
                        #break

                # Add summary data for current wsi to a list, and save to CSV file
                summary_arr = [list_of_region_dicts_in_current_wsi[0]['wsi_filename'], n_tiles_in_current_wsi_before_aug, n_tiles_in_current_wsi_after_aug,
                               DESIRED_TILES_IN_EACH_WSI, current_augment_multiplier, temp_aug_multiplier]

                # Write result to summary file
                try:
                    with open(current_mode_summary_path + TILES_IN_WSI_SUMMARY_CSV_FILE, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(summary_arr)
                except Exception as e:
                    my_functions.my_print('Error writing to file', error=True)
                    my_functions.my_print(e, error=True)

            # Balance dataset if enabled
            if BALANCE_DATASET_BY_AUGMENTATION:
                for n in range(N_CLASSES_TRAINING):
                    i = 0
                    augment_argument = 1

                    # Calculate size of each class, used to balance dataset
                    n_tile_train_class0 = sum(1 for _, tile in training_dict.items() if tile[current_label_to_use] == 0)
                    n_tile_train_class1 = sum(1 for _, tile in training_dict.items() if tile[current_label_to_use] == 1)
                    n_tile_train_class2 = sum(1 for _, tile in training_dict.items() if tile[current_label_to_use] == 2)
                    n_tile_train_class3 = sum(1 for _, tile in training_dict.items() if tile[current_label_to_use] == 3)
                    list_of_sizes = [n_tile_train_class0, n_tile_train_class1, n_tile_train_class2, n_tile_train_class3]

                    # Stop if all classes are balanced
                    if n_tile_train_class0 * N_CLASSES_TRAINING == sum(list_of_sizes):
                        break

                    # Print size of dataset before augmentation
                    print('Balancing dataset - iteration {}'.format(n))
                    for p in range(N_CLASSES_TRAINING):
                        print('Class {} - Size {}'.format(p, list_of_sizes[p]))

                    # Find the smallest class to augment
                    if (max(n_tile_train_class0, n_tile_train_class1, n_tile_train_class2, n_tile_train_class3) - n_tile_train_class0) > 0:
                        aug_size = max(n_tile_train_class0, n_tile_train_class1, n_tile_train_class2, n_tile_train_class3) - n_tile_train_class0
                        class_to_augment = 0
                    elif (max(n_tile_train_class0, n_tile_train_class1, n_tile_train_class2, n_tile_train_class3) - n_tile_train_class1) > 0:
                        aug_size = max(n_tile_train_class0, n_tile_train_class1, n_tile_train_class2, n_tile_train_class3) - n_tile_train_class1
                        class_to_augment = 1
                    elif (max(n_tile_train_class0, n_tile_train_class1, n_tile_train_class2, n_tile_train_class3) - n_tile_train_class2) > 0:
                        aug_size = max(n_tile_train_class0, n_tile_train_class1, n_tile_train_class2, n_tile_train_class3) - n_tile_train_class2
                        class_to_augment = 2
                    elif (max(n_tile_train_class0, n_tile_train_class1, n_tile_train_class2, n_tile_train_class3) - n_tile_train_class3) > 0:
                        aug_size = max(n_tile_train_class0, n_tile_train_class1, n_tile_train_class2, n_tile_train_class3) - n_tile_train_class3
                        class_to_augment = 3

                    # Augment dataset until balanced. Random tiles are extracted, and the augment argument is changed.
                    while i < aug_size:
                        augment_argument += 1
                        for _, value in sorted(training_dict.items(), key=lambda x: random.random()):
                            if value[current_label_to_use] == class_to_augment:
                                temp_aug_value = value.copy()
                                temp_aug_value['augmentarg'] = augment_argument
                                training_dict[len(training_dict)] = temp_aug_value
                                i += 1
                                # Stop augmenting when condition is met
                                if i == aug_size:
                                    break

            # Get validation data. Go through each patient and extract tiles
            for current_wsi in wsi_list_validation:
                # Restore coordinate data
                list_of_region_dicts_in_current_wsi = my_functions.pickle_load(DIAGNOSTIC_VALIDATION_DICTS_PATH + current_wsi)
                """
                Example of list_of_region_dicts_in_current_wsi:
                    [{  'tissue_type': 'N/A', 
                        'wsi_filename': 'H3374-10A HES_2015-08-20 10_12_19.scn', 
                        'WHO73': 0, 
                        'Recurrence': 1, 
                        'Stage': 0, 
                        'Progression': 0, 
                        'WHO04': 1,  
                        'tile_dict': {0: {'coordinates_400x': (162399, 47262), 'coordinates_100x': (40502, 11718), 'coordinates_25x': (10029, 2831)}, 1:{...}, 2: ... } }]
                """

                # Add data to main dict for counting and printing dataset size
                wsi_class_counter_validation.append(list_of_region_dicts_in_current_wsi[0][current_label_to_use])

                for current_region in list_of_region_dicts_in_current_wsi:

                    for tile_index, value in current_region['tile_dict'].items():
                        """
                        Example of value:
                            'value': {
                            0: {
                                'coordinates_400x': (162399, 47262), 
                                'coordinates_100x': (40502, 11718), 
                                'coordinates_25x': (10029, 2831)
                            }, 
                            1:{...}, 2: ... } }
                        """

                        # No augmentation, add to dict.
                        tempvalue = value.copy()
                        tempvalue['augmentarg'] = 1
                        tempvalue['path'] = SCN_PATH + current_region['wsi_filename'].split(".")[0] + '/' + current_region['wsi_filename']
                        tempvalue['WHO73'] = current_region['WHO73']
                        tempvalue['Recurrence'] = current_region['Recurrence']
                        tempvalue['Stage'] = current_region['Stage']
                        tempvalue['Progression'] = current_region['Progression']
                        tempvalue['WHO04'] = current_region['WHO04']
                        validation_dict[len(validation_dict)] = tempvalue

                        # break out from regions (for debugging)
                        if SMALL_DATASET_DEBUG_MODE is True and tile_index >= 512:
                            break

                    # break out from regions (for debugging)
                    #if SMALL_DATASET_DEBUG_MODE is True and len(validation_dict) >= 512:
                        #break

            # Calculate size of dataset
            train_dataset_size = len(training_dict)
            validation_dataset_size = len(validation_dict)
            percentage_train = train_dataset_size / (train_dataset_size + validation_dataset_size) * 100
            percentage_val = validation_dataset_size / (train_dataset_size + validation_dataset_size) * 100
            # endregion

            # region CALCULATE SAMPLES OF EACH CLASS AND PRINT OUT DATASET
            # Count how many training samples we have of each class
            if current_label_to_use in ['Recurrence', 'Progression', 'Stage', 'WHO04']:
                ##### WSI TRAINING #####
                n_wsi_train_class0 = sum(1 for n in wsi_class_counter_training if n == 0)
                n_wsi_train_class1 = sum(1 for n in wsi_class_counter_training if n == 1)
                percentage_wsi_train_class0 = n_wsi_train_class0 / (n_wsi_train_class0 + n_wsi_train_class1) * 100
                percentage_wsi_train_class1 = n_wsi_train_class1 / (n_wsi_train_class0 + n_wsi_train_class1) * 100
                ##### TILES TRAINING #####
                n_tile_train_class0 = sum(1 for _, tile in training_dict.items() if tile[current_label_to_use] == 0)
                n_tile_train_class1 = sum(1 for _, tile in training_dict.items() if tile[current_label_to_use] == 1)
                percentage_tile_class0 = n_tile_train_class0 / (n_tile_train_class0 + n_tile_train_class1) * 100
                percentage_tile_class1 = n_tile_train_class1 / (n_tile_train_class0 + n_tile_train_class1) * 100
                ##### WSI VALIDATION #####
                n_wsi_val_class0 = sum(1 for n in wsi_class_counter_validation if n == 0)
                n_wsi_val_class1 = sum(1 for n in wsi_class_counter_validation if n == 1)
                percentage_wsi_val_class0 = n_wsi_val_class0 / (n_wsi_val_class0 + n_wsi_val_class1) * 100
                percentage_wsi_val_class1 = n_wsi_val_class1 / (n_wsi_val_class0 + n_wsi_val_class1) * 100
                ##### TILES VALIDATION #####
                n_tile_val_class0 = sum(1 for _, tile in validation_dict.items() if tile[current_label_to_use] == 0)
                n_tile_val_class1 = sum(1 for _, tile in validation_dict.items() if tile[current_label_to_use] == 1)
                percentage_tile_val_class0 = n_tile_val_class0 / (n_tile_val_class0 + n_tile_val_class1) * 100
                percentage_tile_val_class1 = n_tile_val_class1 / (n_tile_val_class0 + n_tile_val_class1) * 100
            elif current_label_to_use == 'WHO73':
                ##### WSI TRAINING #####
                n_wsi_train_class0 = sum(1 for n in wsi_class_counter_training if n == 0)
                n_wsi_train_class1 = sum(1 for n in wsi_class_counter_training if n == 1)
                n_wsi_train_class2 = sum(1 for n in wsi_class_counter_training if n == 2)
                percentage_wsi_train_class0 = n_wsi_train_class0 / (n_wsi_train_class0 + n_wsi_train_class1 + n_wsi_train_class2) * 100
                percentage_wsi_train_class1 = n_wsi_train_class1 / (n_wsi_train_class0 + n_wsi_train_class1 + n_wsi_train_class2) * 100
                percentage_wsi_train_class2 = n_wsi_train_class2 / (n_wsi_train_class0 + n_wsi_train_class1 + n_wsi_train_class2) * 100
                ##### TILES TRAINING #####
                n_tile_train_class0 = sum(1 for _, tile in training_dict.items() if tile[current_label_to_use] == 0)
                n_tile_train_class1 = sum(1 for _, tile in training_dict.items() if tile[current_label_to_use] == 1)
                n_tile_train_class2 = sum(1 for _, tile in training_dict.items() if tile[current_label_to_use] == 2)
                percentage_tile_class0 = n_tile_train_class0 / (n_tile_train_class0 + n_tile_train_class1 + n_tile_train_class2) * 100
                percentage_tile_class1 = n_tile_train_class1 / (n_tile_train_class0 + n_tile_train_class1 + n_tile_train_class2) * 100
                percentage_tile_class2 = n_tile_train_class2 / (n_tile_train_class0 + n_tile_train_class1 + n_tile_train_class2) * 100
                ##### WSI VALIDATION #####
                n_wsi_val_class0 = sum(1 for n in wsi_class_counter_validation if n == 0)
                n_wsi_val_class1 = sum(1 for n in wsi_class_counter_validation if n == 1)
                n_wsi_val_class2 = sum(1 for n in wsi_class_counter_validation if n == 2)
                percentage_wsi_val_class0 = n_wsi_val_class0 / (n_wsi_val_class0 + n_wsi_val_class1 + n_wsi_val_class2) * 100
                percentage_wsi_val_class1 = n_wsi_val_class1 / (n_wsi_val_class0 + n_wsi_val_class1 + n_wsi_val_class2) * 100
                percentage_wsi_val_class2 = n_wsi_val_class2 / (n_wsi_val_class0 + n_wsi_val_class1 + n_wsi_val_class2) * 100
                ##### TILES VALIDATION #####
                n_tile_val_class0 = sum(1 for _, tile in validation_dict.items() if tile[current_label_to_use] == 0)
                n_tile_val_class1 = sum(1 for _, tile in validation_dict.items() if tile[current_label_to_use] == 1)
                n_tile_val_class2 = sum(1 for _, tile in validation_dict.items() if tile[current_label_to_use] == 2)
                percentage_tile_val_class0 = n_tile_val_class0 / (n_tile_val_class0 + n_tile_val_class1 + n_tile_val_class2) * 100
                percentage_tile_val_class1 = n_tile_val_class1 / (n_tile_val_class0 + n_tile_val_class1 + n_tile_val_class2) * 100
                percentage_tile_val_class2 = n_tile_val_class2 / (n_tile_val_class0 + n_tile_val_class1 + n_tile_val_class2) * 100

            # Print out training and validation dataset numbers
            my_functions.my_print('\tTRAINING DATASET')
            my_functions.my_print('\t\tTotal size:\t\t\t\t\t{:,} tiles ({:.1f} %) from {} WSIs'.format(train_dataset_size, percentage_train, len(wsi_list_training)))

            # if current_label_to_use in ['Recurrence', 'Progression', 'Stage', 'WHO04']:
            my_functions.my_print('\t\tClass 0 ({}):\t{:,} tiles ({:.1f} %) from {} WSIs ({:.1f} %)'.format(NAME_OF_CLASSES_ALL_DISPLAYNAME[0], n_tile_train_class0, percentage_tile_class0, n_wsi_train_class0, percentage_wsi_train_class0))
            my_functions.my_print('\t\tClass 1 ({}):\t\t{:,} tiles ({:.1f} %) from {} WSIs ({:.1f} %)'.format(NAME_OF_CLASSES_ALL_DISPLAYNAME[1], n_tile_train_class1, percentage_tile_class1, n_wsi_train_class1, percentage_wsi_train_class1))
            if current_label_to_use == 'WHO73':
                my_functions.my_print('\t\tClass 2 ({}):\t{:,} tiles ({:.1f} %) from {} WSIs ({:.1f} %)'.format(NAME_OF_CLASSES_ALL_DISPLAYNAME[2], n_tile_train_class2, percentage_tile_class2, n_wsi_train_class2, percentage_wsi_train_class2))

            my_functions.my_print('\tVALIDATION DATASET')
            my_functions.my_print('\t\tTotal size:\t\t\t\t\t{:,} tiles ({:.1f} %) from {} WSIs'.format(validation_dataset_size, percentage_val, len(wsi_list_validation)))

            # if current_label_to_use in ['Recurrence', 'Progression', 'Stage', 'WHO04']:
            my_functions.my_print('\t\tClass 0 ({}):\t{:,} tiles ({:.1f} %) from {} WSIs ({:.1f} %)'.format(NAME_OF_CLASSES_ALL_DISPLAYNAME[0], n_tile_val_class0, percentage_tile_val_class0, n_wsi_val_class0, percentage_wsi_val_class0))
            my_functions.my_print('\t\tClass 1 ({}):\t\t{:,} tiles ({:.1f} %) from {} WSIs ({:.1f} %)'.format(NAME_OF_CLASSES_ALL_DISPLAYNAME[1], n_tile_val_class1, percentage_tile_val_class1, n_wsi_val_class1, percentage_wsi_val_class1))
            if current_label_to_use == 'WHO73':
                my_functions.my_print('\t\tClass 2 ({}):\t{:,} tiles ({:.1f} %) from {} WSIs ({:.1f} %)'.format(NAME_OF_CLASSES_ALL_DISPLAYNAME[2], n_tile_val_class2, percentage_tile_val_class2, n_wsi_val_class2, percentage_wsi_val_class2))
            # endregion

            # region CREATE MODEL AND DATA GENERATORS
            if current_model_mode == 'mono':
                current_mode = 'Diagnostic transfer learning (Mono-scale)'
                current_deep_learning_model, current_latent_size = my_functions.get_mono_scale_model(img_width=TILE_SIZE_DIAGNOSTIC,
                                                                                                     img_height=TILE_SIZE_DIAGNOSTIC,
                                                                                                     n_channels=N_CHANNELS,
                                                                                                     N_CLASSES=N_CLASSES_TRAINING,
                                                                                                     base_model=current_base_model,
                                                                                                     layer_config=current_layer_config,
                                                                                                     n_neurons1=current_n_neurons1,
                                                                                                     n_neurons2=current_n_neurons2,
                                                                                                     n_neurons3=current_n_neurons3,
                                                                                                     freeze_base_model=current_freeze_base_model,
                                                                                                     base_model_pooling=current_base_model_pooling,
                                                                                                     dropout=current_dropout)

                train_generator = my_functions.mode_8_mono_coordinates_generator(tile_dicts=training_dict,
                                                                                 batch_size=batch_size,
                                                                                 n_classes=N_CLASSES_TRAINING,
                                                                                 shuffle=True,
                                                                                 TILE_SIZE=TILE_SIZE_DIAGNOSTIC,
                                                                                 which_scale_to_use=current_scale_to_use,
                                                                                 label=current_label_to_use,
                                                                                 base_model=current_base_model,
                                                                                 PRE_DIVIDE_TILES_BY_255=PRE_DIVIDE_TILES_BY_255,
                                                                                 PRE_RGB_TO_BGR=PRE_RGB_TO_BGR,
                                                                                 PRE_SUBTRACT_MEAN_FROM_TILE=PRE_SUBTRACT_MEAN_FROM_TILE)

                validation_generator = my_functions.mode_8_mono_coordinates_generator(tile_dicts=validation_dict,
                                                                                      batch_size=batch_size,
                                                                                      n_classes=N_CLASSES_TRAINING,
                                                                                      shuffle=True,
                                                                                      TILE_SIZE=TILE_SIZE_DIAGNOSTIC,
                                                                                      which_scale_to_use=current_scale_to_use,
                                                                                      label=current_label_to_use,
                                                                                      base_model=current_base_model,
                                                                                      PRE_DIVIDE_TILES_BY_255=PRE_DIVIDE_TILES_BY_255,
                                                                                      PRE_RGB_TO_BGR=PRE_RGB_TO_BGR,
                                                                                      PRE_SUBTRACT_MEAN_FROM_TILE=PRE_SUBTRACT_MEAN_FROM_TILE)
            elif current_model_mode == 'di':
                current_mode = 'Diagnostic transfer learning (Di-scale)'
                current_deep_learning_model, current_latent_size = my_functions.get_di_scale_model(img_width=TILE_SIZE_DIAGNOSTIC,
                                                                                                   img_height=TILE_SIZE_DIAGNOSTIC,
                                                                                                   n_channels=N_CHANNELS,
                                                                                                   N_CLASSES=N_CLASSES_TRAINING,
                                                                                                   base_model=current_base_model,
                                                                                                   layer_config=current_layer_config,
                                                                                                   n_neurons1=current_n_neurons1,
                                                                                                   n_neurons2=current_n_neurons2,
                                                                                                   n_neurons3=current_n_neurons3,
                                                                                                   freeze_base_model=current_freeze_base_model,
                                                                                                   base_model_pooling=current_base_model_pooling,
                                                                                                   dropout=current_dropout)

                train_generator = my_functions.mode_8_di_coordinates_generator(tile_dicts=training_dict,
                                                                               batch_size=batch_size,
                                                                               n_classes=N_CLASSES_TRAINING,
                                                                               shuffle=True,
                                                                               TILE_SIZE=TILE_SIZE_DIAGNOSTIC,
                                                                               which_scale_to_use=current_scale_to_use,
                                                                               label=current_label_to_use,
                                                                               PRE_DIVIDE_TILES_BY_255=PRE_DIVIDE_TILES_BY_255,
                                                                               PRE_RGB_TO_BGR=PRE_RGB_TO_BGR,
                                                                               PRE_SUBTRACT_MEAN_FROM_TILE=PRE_SUBTRACT_MEAN_FROM_TILE)

                validation_generator = my_functions.mode_8_di_coordinates_generator(tile_dicts=validation_dict,
                                                                                    batch_size=batch_size,
                                                                                    n_classes=N_CLASSES_TRAINING,
                                                                                    shuffle=True,
                                                                                    TILE_SIZE=TILE_SIZE_DIAGNOSTIC,
                                                                                    which_scale_to_use=current_scale_to_use,
                                                                                    label=current_label_to_use,
                                                                                    PRE_DIVIDE_TILES_BY_255=PRE_DIVIDE_TILES_BY_255,
                                                                                    PRE_RGB_TO_BGR=PRE_RGB_TO_BGR,
                                                                                    PRE_SUBTRACT_MEAN_FROM_TILE=PRE_SUBTRACT_MEAN_FROM_TILE)
            elif current_model_mode == 'tri':
                current_mode = 'Diagnostic transfer learning (Tri-scale)'
                current_deep_learning_model, current_latent_size = my_functions.get_tri_scale_model(img_width=TILE_SIZE_DIAGNOSTIC,
                                                                                                    img_height=TILE_SIZE_DIAGNOSTIC,
                                                                                                    n_channels=N_CHANNELS,
                                                                                                    N_CLASSES=N_CLASSES_TRAINING,
                                                                                                    base_model=current_base_model,
                                                                                                    layer_config=current_layer_config,
                                                                                                    n_neurons1=current_n_neurons1,
                                                                                                    n_neurons2=current_n_neurons2,
                                                                                                    n_neurons3=current_n_neurons3,
                                                                                                    freeze_base_model=current_freeze_base_model,
                                                                                                    base_model_pooling=current_base_model_pooling,
                                                                                                    dropout=current_dropout)

                train_generator = my_functions.mode_8_tri_coordinates_generator(tile_dicts=training_dict,
                                                                                batch_size=batch_size,
                                                                                n_classes=N_CLASSES_TRAINING,
                                                                                shuffle=True,
                                                                                TILE_SIZE=TILE_SIZE_DIAGNOSTIC,
                                                                                which_scale_to_use=current_scale_to_use,
                                                                                label=current_label_to_use,
                                                                                PRE_DIVIDE_TILES_BY_255=PRE_DIVIDE_TILES_BY_255,
                                                                                PRE_RGB_TO_BGR=PRE_RGB_TO_BGR,
                                                                                PRE_SUBTRACT_MEAN_FROM_TILE=PRE_SUBTRACT_MEAN_FROM_TILE)

                validation_generator = my_functions.mode_8_tri_coordinates_generator(tile_dicts=validation_dict,
                                                                                     batch_size=batch_size,
                                                                                     n_classes=N_CLASSES_TRAINING,
                                                                                     shuffle=True,
                                                                                     TILE_SIZE=TILE_SIZE_DIAGNOSTIC,
                                                                                     which_scale_to_use=current_scale_to_use,
                                                                                     label=current_label_to_use,
                                                                                     PRE_DIVIDE_TILES_BY_255=PRE_DIVIDE_TILES_BY_255,
                                                                                     PRE_RGB_TO_BGR=PRE_RGB_TO_BGR,
                                                                                     PRE_SUBTRACT_MEAN_FROM_TILE=PRE_SUBTRACT_MEAN_FROM_TILE)
            else:
                my_functions.my_print('Error in MULTISCALE_MODE_DTL. Stopping program.', error=True)
                exit()
            # endregion

            # region LOAD MODEL WEIGHTS, CREATE OPTIMIZERS, LOSS FUNCTION AND COMPILE MODEL

            # Calculate compression ratio
            compression = abs(round((1 - (current_latent_size / (TILE_SIZE_DIAGNOSTIC * TILE_SIZE_DIAGNOSTIC * N_CHANNELS))) * 100, 1))

            # Check if there exist some weights we can load. Else, start a new model
            if START_NEW_MODEL in [False, 'False', 'false'] and len(os.listdir(weight_save_path)) >= 1:
                all_weights = os.listdir(weight_save_path)
                all_weights = sorted(all_weights, key=lambda a: int(a.split("_")[1].split(".")[0]))
                last_weight = all_weights[-1]
                weight_filename = weight_save_path + last_weight
                current_deep_learning_model.load_weights(weight_filename)
                my_functions.my_print('Loaded weights: {}'.format(weight_filename))
                # noinspection PyTypeChecker
                start_epoch = int(last_weight.split('_')[1].split(".")[0])

                # Restore training data
                pickle_reader = open(current_run_path + METADATA_FOLDER + str(current_model_no) + TRAINING_DATA_DTL_PICKLE_FILE, 'rb')
                (current_epoch, model_duration, current_best_train_loss_data, current_best_train_acc_data,
                 current_best_val_loss_data, current_best_val_acc_data, current_best_val_acc_epoch_data, current_best_val_loss_epoch_data,
                 current_best_train_acc_epoch, current_best_train_loss_epoch, lr_history) = pickle.load(pickle_reader)
                pickle_reader.close()

                # If model was frozen, we need to freeze it now also
                if start_epoch > delayed_unfreeze_start_epoche_vgg16_vgg19:
                    for layer in current_deep_learning_model.layers:
                        if str(layer.name).split("_")[0] in ['block1', 'block2', 'block3', 'block4', 'block5', 'global']:
                            layer.trainable = False
            else:
                # Start new model
                start_epoch = 0
                current_epoch = 0
                current_best_train_loss_data = 50000
                current_best_train_acc_data = 0
                current_best_train_acc_epoch = 0
                current_best_train_loss_epoch = 0
                current_best_val_loss_data = 50000
                current_best_val_acc_data = 0
                current_best_val_acc_epoch_data = 1
                current_best_val_loss_epoch_data = 1
                lr_history = []
                model_duration = 0

                # Backup all data
                pickle_writer = open(current_run_path + METADATA_FOLDER + str(current_model_no) + TRAINING_DATA_DTL_PICKLE_FILE, 'wb')
                pickle.dump(
                    (current_epoch, model_duration, current_best_train_loss_data, current_best_train_acc_data,
                     current_best_val_loss_data, current_best_val_acc_data, current_best_val_acc_epoch_data, current_best_val_loss_epoch_data,
                     current_best_train_acc_epoch, current_best_train_loss_epoch, lr_history), pickle_writer)
                pickle_writer.close()

            # Define optimizer
            if current_optimizer == 'adam':
                my_optimist = Adam(lr=current_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            elif current_optimizer == 'adamax':
                my_optimist = Adamax(lr=current_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            elif current_optimizer == 'nadam':
                my_optimist = Nadam(lr=current_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
            elif current_optimizer == 'adadelta':
                my_optimist = Adadelta(lr=current_learning_rate, rho=0.95, epsilon=None, decay=0.0)
            elif current_optimizer == 'adagrad':
                my_optimist = Adagrad(lr=current_learning_rate, epsilon=None, decay=0.0)
            elif current_optimizer == 'SGD':
                my_optimist = SGD(lr=current_learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
            else:
                my_optimist = None
                my_functions.my_print('Error on choosing optimizer. stopping program', error=True)
                exit()

            # Compile the model
            current_deep_learning_model.compile(optimizer=my_optimist, loss='categorical_crossentropy', metrics=['accuracy'])

            # Save model structure to file
            with open(current_mode_summary_path + current_model_name + '/arcitecture_report.txt', 'w') as fh:
                # Pass the file handle in as a lambda function to make it callable
                current_deep_learning_model.summary(print_fn=lambda x: fh.write(x + '\n'))

            # Plot model
            try:
                plot_model(current_deep_learning_model, to_file=current_mode_summary_path + current_model_name + '/model.png', show_shapes=True, show_layer_names=True)
            except NameError:
                my_functions.my_print('Failed to use plot_model.')

            # Get the number of parameters in the model
            n_trainable_parameters_start = int(np.sum([K.count_params(p) for p in set(current_deep_learning_model.trainable_weights)]))
            n_non_trainable_parameters_start = int(np.sum([K.count_params(p) for p in set(current_deep_learning_model.non_trainable_weights)]))
            # endregion

            # region CALLBACK FUNCTIONS
            my_early_stop_loss = EarlyStopping(monitor='val_loss',  # quantity to be monitored
                                               min_delta=0.0001,  # minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
                                               patience=EARLY_STOPPING_PATIENCE,  # number of epochs with no improvement after which training will be stopped.
                                               verbose=1,  # verbosity mode.
                                               mode='auto',
                                               baseline=None,
                                               restore_best_weights=True)

            # factor: factor by which the learning rate will be reduced. new_lr = lr * factor
            my_reduce_LR_callback = ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.1,
                                                      patience=REDUCE_LR_PATIENCE,
                                                      verbose=1,
                                                      mode='auto',
                                                      min_delta=0.0001,
                                                      cooldown=0,
                                                      min_lr=0.00000001)

            # Define what to do every N epochs
            class MyCallbackFunction(Callback):
                def __init__(self, model, current_epoch, model_duration, current_best_train_loss_data, current_best_train_acc_data,
                             current_best_val_loss_data, current_best_val_acc_data, current_best_val_acc_epoch_data,
                             current_best_val_loss_epoch_data, current_best_train_acc_epoch,
                             current_best_train_loss_epoch, n_trainable_parameters_end, n_non_trainable_parameters_end, lr_history):
                    super().__init__()
                    self.model = model
                    self.current_epoch = current_epoch
                    self.model_duration = model_duration
                    self.current_best_train_loss = current_best_train_loss_data
                    self.current_best_train_acc = current_best_train_acc_data
                    self.current_best_train_loss_epoch = current_best_train_loss_epoch
                    self.current_best_train_acc_epoch = current_best_train_acc_epoch
                    self.current_best_val_loss = current_best_val_loss_data
                    self.current_best_val_acc = current_best_val_acc_data
                    self.current_best_val_acc_epoch = current_best_val_acc_epoch_data
                    self.current_best_val_loss_epoch = current_best_val_loss_epoch_data
                    self.n_trainable_parameters_end = n_trainable_parameters_end
                    self.n_non_trainable_parameters_end = n_non_trainable_parameters_end
                    self.lr_history_list = lr_history
                    self.epoch_timer = 0

                def on_epoch_begin(self, epoch, logs=None):
                    # Raise the epoch number
                    self.current_epoch += 1

                    # Start epoch times
                    self.epoch_timer = time.time()

                    # Print epoch number to log
                    my_functions.my_print('Starting epoch no. {}'.format(self.current_epoch), visible=False)

                    # If we are using VGG16/VGG19, we need to check if we should unfreeze some layers
                    if current_base_model in ['VGG16', 'VGG19'] and current_freeze_base_model == 'Hybrid':

                        # When number of epoches is equal to "delayed start", go through each layer en unfreeze them
                        if self.current_epoch == delayed_unfreeze_start_epoche_vgg16_vgg19:
                            my_functions.my_print('Unfreezing layers in model')
                            for layer in current_deep_learning_model.layers:
                                if str(layer.name).split("_")[0] in blocks_to_unfreeze_vgg16_vgg19:
                                    layer.trainable = True

                            # Re-compile model
                            current_deep_learning_model.compile(optimizer=my_optimist,
                                                                loss='categorical_crossentropy',  # binary_crossentropy / categorical_crossentropy / sparse_categorical_crossentropy
                                                                metrics=['accuracy'])  # categorical_accuracy / binary_accuracy

                            # Print each layer and trainable status (for logging)
                            my_layers = [(layer, layer.name, layer.trainable) for layer in current_deep_learning_model.layers]
                            for n in my_layers:
                                my_functions.my_print(n, visible=False)

                            # Get the number of parameters in the model
                            self.n_trainable_parameters_end = int(np.sum([K.count_params(p) for p in set(current_deep_learning_model.trainable_weights)]))
                            self.n_non_trainable_parameters_end = int(np.sum([K.count_params(p) for p in set(current_deep_learning_model.non_trainable_weights)]))

                def on_epoch_end(self, batch, logs=None):

                    # Print learning rate and save learning rate to list
                    start_lr = self.model.optimizer.lr
                    decay = self.model.optimizer.decay
                    iterations = self.model.optimizer.iterations
                    lr_with_decay = start_lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
                    # print(K.eval(start_lr))
                    # print(K.eval(decay))
                    # print(K.eval(iterations))
                    print('lr_with_decay: ', K.eval(lr_with_decay))
                    self.lr_history_list.append(K.eval(lr_with_decay))

                    # Save weights
                    weight_save_filename = weight_save_path + 'Epoch_' + str(self.current_epoch) + '.h5'
                    self.model.save_weights(weight_save_filename)

                    # If new best model, save the accuracy of the model
                    if logs.get('loss') < self.current_best_train_loss:
                        self.current_best_train_loss = logs.get('loss')
                        self.current_best_train_loss_epoch = self.current_epoch
                    if logs.get('acc') > self.current_best_train_acc:
                        self.current_best_train_acc = logs.get('acc')
                        self.current_best_train_acc_epoch = self.current_epoch
                    if logs.get('val_loss') < self.current_best_val_loss:
                        self.current_best_val_loss = logs.get('val_loss')
                        self.current_best_val_loss_epoch = self.current_epoch
                    if logs.get('val_acc') > self.current_best_val_acc:
                        self.current_best_val_acc = logs.get('val_acc')
                        self.current_best_val_acc_epoch = self.current_epoch

                    # Delete previous model (to save HDD space)
                    for previous_epochs in range(self.current_epoch - 1):
                        if not (previous_epochs == self.current_best_val_acc_epoch) and not (previous_epochs == self.current_best_val_loss_epoch):
                            delete_filename = '/Epoch_{}.*'.format(previous_epochs)
                            for files in glob.glob(weight_save_path + delete_filename):
                                os.remove(files)

                    # Calculate current epoch time, and append to total time
                    self.model_duration += (time.time() - self.epoch_timer)

                    # Update number of epoches trained for current model
                    current_model_dict['trained_epoches'] = self.current_epoch

                    # Save to file
                    my_functions.pickle_save(ALL_MODEL_PARAMETERS, current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE)

                    # Backup all data
                    pickle_writer = open(current_run_path + METADATA_FOLDER + str(current_model_no) + TRAINING_DATA_DTL_PICKLE_FILE, 'wb')
                    pickle.dump(
                        (self.current_epoch, self.model_duration, self.current_best_train_loss, self.current_best_train_acc,
                         self.current_best_val_loss, self.current_best_val_acc, self.current_best_val_acc_epoch, self.current_best_val_loss_epoch,
                         self.current_best_train_acc_epoch, self.current_best_train_loss_epoch, self.lr_history_list), pickle_writer)
                    pickle_writer.close()

                def on_train_begin(self, logs=None):
                    # Print each layer and trainable status (for logging)
                    # my_layers = [(layer, layer.name, layer.trainable) for layer in current_deep_learning_model.layers]
                    # for n in my_layers:
                    #     my_functions.my_print(n, visible=False)
                    pass

                def on_train_end(self, logs=None):

                    # Do a final evaluation of the validation dataset and save confusion matrix, classification report and misclassified images
                    if SAVE_CONF_MAT_AND_REPORT_ON_END is True:
                        my_functions.my_print('Final evaluation of validation dataset')

                        # Define some lists. If already defined, they will be reset for each epoch.
                        y_true_class_total = []
                        y_pred_class_total = []

                        # Calculate number of steps
                        if float(validation_dataset_size / batch_size).is_integer():
                            n_steps_val = int(np.floor(validation_dataset_size / batch_size))
                        else:
                            n_steps_val = int(np.floor(validation_dataset_size / batch_size)) + 1

                        # MONO-SCALE
                        if current_model_mode == 'mono':
                            # Go through all batches of validation images
                            for batch_index in range(n_steps_val):
                                # Load one batch of images and labels
                                y_images_1, y_true_one_hot_encoded = validation_generator.__getitem__(batch_index)

                                # Use model to predict images
                                y_pred_probabilities = self.model.predict(x=y_images_1,
                                                                          batch_size=None,
                                                                          verbose=0,
                                                                          steps=None)

                                # Convert variables, append values for all batched
                                y_true_class_total.extend(np.argmax(y_true_one_hot_encoded, axis=1))
                                y_pred_class_total.extend(np.argmax(y_pred_probabilities, axis=1))

                        # DI-SCALE
                        elif current_model_mode == 'di':
                            # Go through all batches of validation images
                            for batch_index in range(n_steps_val):
                                # Load one batch of images and labels
                                [y_images_1, y_images_2], y_true_one_hot_encoded = validation_generator.__getitem__(batch_index)

                                # Use model to predict images
                                y_pred_probabilities = self.model.predict(x=[y_images_1, y_images_2],
                                                                          batch_size=None,
                                                                          verbose=0,
                                                                          steps=None)

                                # Convert variables, append values for all batched
                                y_true_class_total.extend(np.argmax(y_true_one_hot_encoded, axis=1))
                                y_pred_class_total.extend(np.argmax(y_pred_probabilities, axis=1))


                        # TRI-SCALE
                        elif current_model_mode == 'tri':
                            # Go through all batches of validation images
                            for batch_index in range(n_steps_val):
                                # Load one batch of images and labels
                                [y_images_1, y_images_2, y_images_3], y_true_one_hot_encoded = validation_generator.__getitem__(batch_index)

                                # Use model to predict images
                                y_pred_probabilities = self.model.predict(x=[y_images_1, y_images_2, y_images_3],
                                                                          batch_size=None,
                                                                          verbose=0,
                                                                          steps=None)

                                # Convert variables, append values for all batched
                                y_true_class_total.extend(np.argmax(y_true_one_hot_encoded, axis=1))
                                y_pred_class_total.extend(np.argmax(y_pred_probabilities, axis=1))

                        # Compute confusion matrix
                        cm = confusion_matrix(y_true=y_true_class_total,
                                              y_pred=y_pred_class_total,
                                              labels=cm_label,
                                              sample_weight=None)

                        # Define a title
                        cm_title = '{} - Validation set - Epoch {}'.format(current_label_to_use, self.current_epoch)

                        # Save confusion matrix
                        my_functions.plot_confusion_matrix(cm=cm,
                                                           epoch=self.current_epoch,
                                                           classes=NAME_OF_CLASSES_TRAINING_DISPLAYNAME,
                                                           SUMMARY_PATH=current_mode_summary_path + current_model_name,
                                                           folder_name='Confusion_matrix_validation',
                                                           title=cm_title)

                        cm = np.round(cm / cm.sum(axis=1, keepdims=True), 3)

                        # Save confusion matrix (normalized)
                        my_functions.plot_confusion_matrix(cm=cm,
                                                           epoch='{}_normalized'.format(self.current_epoch),
                                                           classes=NAME_OF_CLASSES_TRAINING_DISPLAYNAME,
                                                           SUMMARY_PATH=current_mode_summary_path + current_model_name,
                                                           folder_name='Confusion_matrix_validation',
                                                           title=cm_title)

                        # Compute classification report
                        cr = classification_report(y_true=y_true_class_total,
                                                   y_pred=y_pred_class_total,
                                                   target_names=NAME_OF_CLASSES_TRAINING,
                                                   digits=8)

                        # Parse the classification report, so we can save it to a CSV file
                        tmp = list()
                        for row in cr.split("\n"):
                            parsed_row = [x for x in row.split(" ") if len(x) > 0]
                            if len(parsed_row) > 0:
                                tmp.append(parsed_row)

                        # Add an empty item to line up header in CSV file
                        tmp[0].insert(0, '')

                        # Save classification report to CSV
                        with open(current_model_classification_report_path + 'Validation_classification_report_' + str(self.current_epoch) + '.csv', 'w') as newFile:
                            newFileWriter = csv.writer(newFile, delimiter=';', lineterminator='\r', quoting=csv.QUOTE_MINIMAL)
                            for rows in range(len(tmp)):
                                newFileWriter.writerow(tmp[rows])

                    # Check if training was stopped by early stopping callback
                    if my_early_stop_loss.stopped_epoch > 0:
                        # Update early stopping for current model
                        current_model_dict['early_stopping'] = my_early_stop_loss.stopped_epoch + 1

                    # Save learning rate history plot
                    my_functions.save_learning_rate_history_plot(history=self.lr_history_list,
                                                                 path=current_mode_summary_path + current_model_name,
                                                                 mode=current_mode,
                                                                 model_no=current_model_no)

                    # Calculate training time
                    m, s = divmod(self.model_duration, 60)
                    h, m = divmod(m, 60)
                    model_time = '%02d:%02d:%02d' % (h, m, s)

                    # Write result to summary.csv file
                    my_functions.summary_csv_file_update(SUMMARY_CSV_FILE_PATH=SUMMARY_CSV_FILE_PATH,
                                                         MODE='8a',
                                                         model_name=current_model_name,
                                                         label=current_label_to_use,
                                                         base_model=current_base_model,
                                                         freeze_base_model=current_freeze_base_model,
                                                         blocks_to_unfreeze_vgg16_vgg19=blocks_to_unfreeze_vgg16_vgg19,
                                                         delayed_unfreeze_start_epoche_vgg16_vgg19=delayed_unfreeze_start_epoche_vgg16_vgg19,
                                                         base_model_pooling=current_base_model_pooling,
                                                         training_samples=train_dataset_size,
                                                         validation_samples=validation_dataset_size,
                                                         test_samples='N/A',
                                                         layer_config=current_layer_config,
                                                         augment_classes='N/A',
                                                         augment_multiplier='N/A',
                                                         learning_rate=current_learning_rate,
                                                         batch_size=batch_size,
                                                         n_neurons1=current_n_neurons1,
                                                         n_neurons2=current_n_neurons2,
                                                         n_neurons3=current_n_neurons3,
                                                         EARLY_STOPPING_PATIENCE=EARLY_STOPPING_PATIENCE,
                                                         dropout=current_dropout,
                                                         best_train_loss=self.current_best_train_loss,
                                                         best_train_acc=self.current_best_train_acc,
                                                         best_val_loss=self.current_best_val_loss,
                                                         best_val_acc=self.current_best_val_acc,
                                                         best_val_loss_epoch=self.current_best_val_loss_epoch,
                                                         best_val_acc_epoch=self.current_best_val_acc_epoch,
                                                         trained_epoches=self.current_epoch,
                                                         total_epochs=EPOCHES,
                                                         latent_size=current_latent_size,
                                                         compression=compression,
                                                         model_time=model_time,
                                                         optimizer=current_optimizer,
                                                         ReduceLRstatus=ReduceLRstatus,
                                                         n_trainable_parameters_start=n_trainable_parameters_start,
                                                         n_non_trainable_parameters_start=n_non_trainable_parameters_start,
                                                         n_trainable_parameters_end=self.n_trainable_parameters_end,
                                                         n_non_trainable_parameters_end=self.n_non_trainable_parameters_end,
                                                         python_version=sys.version.split(" ")[0],
                                                         keras_version=keras.__version__,
                                                         tf_version=tf_version,
                                                         tile_size=TILE_SIZE_DIAGNOSTIC)

                    # Save files
                    MODELS_AND_LOSS_ARRAY.update({current_model_dict['ID']: self.current_best_val_loss})
                    my_functions.pickle_save(MODELS_AND_LOSS_ARRAY, current_run_path + METADATA_FOLDER + MODELS_AND_LOSS_ARRAY_DTL_PICKLE_FILE)
                    my_functions.pickle_save(ALL_MODEL_PARAMETERS, current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE)

                def on_batch_begin(self, batch, logs=None):
                    pass

                def on_batch_end(self, batch, logs=None):
                    pass

            # Define the callback function array
            main_callback = MyCallbackFunction(model=current_deep_learning_model,
                                               current_epoch=current_epoch,
                                               model_duration=model_duration,
                                               current_best_train_loss_data=current_best_train_loss_data,
                                               current_best_train_acc_data=current_best_train_acc_data,
                                               current_best_val_loss_data=current_best_val_loss_data,
                                               current_best_val_acc_data=current_best_val_acc_data,
                                               current_best_val_acc_epoch_data=current_best_val_acc_epoch_data,
                                               current_best_val_loss_epoch_data=current_best_val_loss_epoch_data,
                                               current_best_train_loss_epoch=current_best_train_loss_epoch,
                                               current_best_train_acc_epoch=current_best_train_acc_epoch,
                                               n_trainable_parameters_end=n_trainable_parameters_start,
                                               n_non_trainable_parameters_end=n_non_trainable_parameters_start,
                                               lr_history=lr_history)

            # Define the callback function array
            callback_array = [csv_logger, main_callback, my_early_stop_loss]

            if ACTIVATE_ReduceLROnPlateau is True:
                callback_array.append(my_reduce_LR_callback)
            # endregion

            # region TRAIN MODEL
            history_DTL_obj = current_deep_learning_model.fit_generator(generator=train_generator,
                                                                        steps_per_epoch=None,
                                                                        epochs=EPOCHES,
                                                                        verbose=1,
                                                                        callbacks=callback_array,
                                                                        validation_data=validation_generator,
                                                                        validation_steps=None,
                                                                        class_weight=None,
                                                                        max_queue_size=MAX_QUEUE_SIZE,
                                                                        workers=N_WORKERS,
                                                                        use_multiprocessing=USE_MULTIPROCESSING,
                                                                        shuffle=True,
                                                                        initial_epoch=start_epoch)

            my_functions.save_history_plot(history=history_DTL_obj,
                                           path=current_mode_summary_path + current_model_name,
                                           mode=current_mode,
                                           model_no=current_model_no)

            # endregion

    my_functions.my_print('Mode 8a finished.')
