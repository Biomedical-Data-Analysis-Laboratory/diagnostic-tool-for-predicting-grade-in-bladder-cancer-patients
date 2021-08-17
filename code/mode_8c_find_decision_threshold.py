from sklearn.metrics import accuracy_score
from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.optimizers import Adamax
from keras.optimizers import Nadam
from keras.optimizers import Adam
from keras.optimizers import SGD
import my_functions
import my_constants
import numpy as np
import operator
import datetime
import pickle
import time
import csv
import os


def predict_validation_dataset(current_run_path, METADATA_FOLDER, MODEL_WEIGHT_FOLDER, N_CHANNELS, WHAT_MODEL_TO_LOAD, WHAT_MODEL_EPOCH_TO_LOAD,
                               MODE_FOLDER, DIAGNOSTIC_TRAINING_DICTS_PATH, SCN_PATH, MODELS_AND_LOSS_ARRAY_DTL_PICKLE_FILE,
                               ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE, TRAINING_DATA_DTL_PICKLE_FILE, TILE_SIZE_DIAGNOSTIC,
                               current_wsi_coordinate_pickle_filename, wsi_filename_no_extension, DEBUG_MODE, MAX_QUEUE_SIZE,
                               N_WORKERS, USE_MULTIPROCESSING, N_REGIONS_TO_PROCESS_DEBUG_MODE, PRE_DIVIDE_TILES_BY_255,
                               PRE_RGB_TO_BGR, PRE_SUBTRACT_MEAN_FROM_TILE, RUN_NEW_PREDICTION):
    # region FILE INIT

    # Start timer
    current_start_time = time.time()

    # Large batch size always gives error message on test-set. Reduce batch size
    batch_size = 16

    # endregion

    # region FIND CORRECT MODE
    # Load model arrays
    try:
        # Restore from file
        MODELS_AND_LOSS_ARRAY = my_functions.pickle_load(current_run_path + METADATA_FOLDER + MODELS_AND_LOSS_ARRAY_DTL_PICKLE_FILE)
        ALL_MODEL_PARAMETERS = my_functions.pickle_load(current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE)
    except NameError:
        my_functions.my_print('No models found, stopping program', error=True)
        exit()

    if WHAT_MODEL_TO_LOAD in ['Best', 'best']:
        # Load best model. Sort all transfer learning models by lowest validation loss, and choose best model.
        MODELS_AND_LOSS_ARRAY_SORTED = sorted(MODELS_AND_LOSS_ARRAY.items(), key=operator.itemgetter(1), reverse=False)
        MODEL_TO_USE = MODELS_AND_LOSS_ARRAY_SORTED[0][0]
    elif isinstance(WHAT_MODEL_TO_LOAD, int):
        # Load specific model
        MODEL_TO_USE = WHAT_MODEL_TO_LOAD
    else:
        my_functions.my_print('Error in WHAT_MODEL_TO_LOAD. stopping program', error=True)
        exit()

    # endregion

    # Loop through models until the model we want
    for current_model_dict in ALL_MODEL_PARAMETERS:

        # Check if current_model is the model we want to use
        if current_model_dict['ID'] == MODEL_TO_USE:

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
            N_CLASSES_TRAINING = sum(1 for _, tile in name_and_index_of_classes.items() if tile['used_in_training'] == 1)
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

            # Define name of the model
            if current_model_mode == 'mono':
                current_model_name = 'Diagnostic_Model_' + str(current_model_no) + '_' + current_model_mode + '_' + current_scale_to_use + '/'
            elif str(current_model_mode) == 'di':
                current_model_name = 'Diagnostic_Model_' + str(current_model_no) + '_' + current_model_mode + '_' + current_scale_to_use[0] + '_' + current_scale_to_use[1] + '/'
            elif current_model_mode == 'tri':
                current_model_name = 'Diagnostic_Model_' + str(current_model_no) + '_' + current_model_mode + '_25x_100x_400x/'

            my_functions.my_print(
                'Testing model on diagnostic validation set - {}, Freeze:{}, Layer_{}, optimizer:{}, learning rate:{}, '
                'batch size:{}, n_neurons1:{}, n_neurons2:{}, n_neurons3:{}, dropout:{}'.format(
                    current_model_name, current_freeze_base_model, current_layer_config, current_optimizer,
                    current_learning_rate, batch_size, current_n_neurons1, current_n_neurons2,
                    current_n_neurons3, current_dropout))

            # Define current_mode_summary_path
            current_mode_summary_path = current_run_path + MODE_FOLDER
            os.makedirs(current_mode_summary_path, exist_ok=True)

            # Make folder for new model.
            current_model_summary_path = current_mode_summary_path + current_model_name + '/'
            os.makedirs(current_model_summary_path, exist_ok=True)

            # Define summary files
            model_prediction_summary_pickle_file = current_model_summary_path + 'model_prediction_summary.obj'
            current_model_summary_csv_file = current_model_summary_path + 'Validation_summary.csv'

            # Make folder for current WSI
            current_wsi_summary_path = current_model_summary_path + wsi_filename_no_extension + '/'
            os.makedirs(current_wsi_summary_path, exist_ok=True)

            # Make a vector with name of each class
            name_of_classes_array = []
            for index, class_name in enumerate(NAME_OF_CLASSES_TRAINING):
                name_of_classes_array.append('Class ' + str(index) + ': ' + class_name)

            # Make an array for top row in summary CSV file
            temp_csv_classes = []
            for name_of_class in NAME_OF_CLASSES_TRAINING:
                temp_csv_classes.append(name_of_class + '_AVG_PROB')
            csv_array = ['IMAGE_NAME', 'TRUE']
            csv_array.extend(temp_csv_classes)
            csv_array.append('WHAT_MODEL_TO_LOAD')
            csv_array.append('WHAT_MODEL_EPOCH_TO_LOAD')
            csv_array.append('DEBUG_MODE')
            csv_array.append('N_REGIONS_TO_PROCESS_DEBUG_MODE')
            csv_array.append('Duration(H:M:S)')
            csv_array.append('Test_performed')

            # Create a new summary.csv file for current test
            if not os.path.isfile(current_model_summary_csv_file):
                try:
                    with open(current_model_summary_csv_file, 'w') as csvfile:
                        csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(csv_array)
                except Exception as e:
                    my_functions.my_print('Error writing to file', error=True)
                    my_functions.my_print(e, error=True)

            # endregion

            # region CREATE MODEL, LOAD WEIGHTS, COMPILE MODEL
            # Create model
            if current_model_mode == 'mono':
                current_deep_learning_model, current_latent_size = my_functions.get_mono_scale_model(
                    img_width=TILE_SIZE_DIAGNOSTIC,
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
            elif current_model_mode == 'di':
                current_deep_learning_model, current_latent_size = my_functions.get_di_scale_model(
                    img_width=TILE_SIZE_DIAGNOSTIC,
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
            elif current_model_mode == 'tri':
                current_deep_learning_model, current_latent_size = my_functions.get_tri_scale_model(
                    img_width=TILE_SIZE_DIAGNOSTIC,
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

            # Define path where weights are saved
            weight_save_path_to_load = current_run_path + MODEL_WEIGHT_FOLDER + current_model_name

            # Check if there exist some weights we can load.
            if len(os.listdir(weight_save_path_to_load)) >= 1:
                # Load weights into model
                if WHAT_MODEL_EPOCH_TO_LOAD in ['Last', 'last']:
                    # Load weights from last epoch
                    all_weights = os.listdir(weight_save_path_to_load)
                    all_weights = sorted(all_weights, key=lambda a: int(a.split("_")[1].split(".")[0]))
                    weight_to_load = all_weights[-1]
                elif WHAT_MODEL_EPOCH_TO_LOAD in ['Best', 'best']:
                    # Load weights from best epoch
                    pickle_reader = open(current_run_path + METADATA_FOLDER + str(current_model_no) + TRAINING_DATA_DTL_PICKLE_FILE, 'rb')
                    (current_epoch, model_duration, current_best_train_loss_data, current_best_train_acc_data,
                     current_best_val_loss_data, current_best_val_acc_data, current_best_val_acc_epoch_data,
                     current_best_val_loss_epoch_data,
                     current_best_train_acc_epoch, current_best_train_loss_epoch, lr_history) = pickle.load(pickle_reader)
                    pickle_reader.close()

                    # Save best epoch number
                    weight_to_load = 'Epoch_' + str(current_best_val_acc_epoch_data) + '.h5'
                elif isinstance(WHAT_MODEL_EPOCH_TO_LOAD, int):
                    # Load weights from a specific epoch
                    weight_to_load = 'Epoch_' + str(WHAT_MODEL_EPOCH_TO_LOAD) + '.h5'
                else:
                    my_functions.my_print('Error in WHAT_MODEL_EPOCH_TO_LOAD. stopping program', error=True)
                    exit()

                weight_filename = weight_save_path_to_load + weight_to_load
                my_functions.my_print('\tLoading weights: {}'.format(weight_filename))
                current_deep_learning_model.load_weights(weight_filename)
            else:
                my_functions.my_print('Found no weights to load in folder. stopping program', error=True)
                exit()

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
            # endregion

            y_true_argmax_current_wsi = []
            y_pred_argmax_current_wsi = []
            list_of_all_predictions_class0 = []
            list_of_all_predictions_class1 = []
            list_of_all_predictions_class2 = []
            list_of_all_predictions_class3 = []

            # Load list containing one dict per region
            list_of_region_dicts_in_current_wsi = my_functions.pickle_load(DIAGNOSTIC_TRAINING_DICTS_PATH + current_wsi_coordinate_pickle_filename)
            """
            Example of list_of_region_dicts_in_current_wsi:
                [   'tissue_type': 'N/A', 
                    'wsi_filename': 'H3374-10A HES_2015-08-20 10_12_19.scn', 
                    'WHO73': 0, 
                    'Recurrence': 1, 
                    'Stage': 0, 
                    'Progression': 0, 
                    'WHO04': 1,  
                    'tile_dict': {0: {'coordinates_400x': (162399, 47262), 'coordinates_100x': (40502, 11718), 'coordinates_25x': (10029, 2831)}, 1:{...}, 2: ... } }]
            """

            # Process each region in current WSI
            for region_index, current_region in enumerate(list_of_region_dicts_in_current_wsi):

                if DEBUG_MODE:
                    print('Processing region {} / {} '.format(region_index + 1, min(N_REGIONS_TO_PROCESS_DEBUG_MODE, len(list_of_region_dicts_in_current_wsi))))
                else:
                    print('Processing region {} / {} '.format(region_index + 1, len(list_of_region_dicts_in_current_wsi)))

                # Reset dataset dict
                region_dataset_dict = dict()

                # Extract each tile
                for _, value in current_region['tile_dict'].items():
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

                    # Add tile to dict
                    tempvalue = value.copy()
                    tempvalue['augmentarg'] = 1
                    tempvalue['path'] = SCN_PATH + current_region['wsi_filename'].split(".")[0] + '/' + current_region['wsi_filename']
                    tempvalue['WHO73'] = current_region['WHO73']
                    tempvalue['Recurrence'] = current_region['Recurrence']
                    tempvalue['Stage'] = current_region['Stage']
                    tempvalue['Progression'] = current_region['Progression']
                    tempvalue['WHO04'] = current_region['WHO04']
                    region_dataset_dict[len(region_dataset_dict)] = tempvalue

                # Create data generator
                if current_model_mode == 'mono':
                    # Create data generators
                    region_generator = my_functions.mode_8_mono_coordinates_generator(tile_dicts=region_dataset_dict,
                                                                                      batch_size=batch_size,
                                                                                      n_classes=N_CLASSES_TRAINING,
                                                                                      shuffle=False,
                                                                                      TILE_SIZE=TILE_SIZE_DIAGNOSTIC,
                                                                                      which_scale_to_use=current_scale_to_use,
                                                                                      label=current_label_to_use,
                                                                                      base_model=current_base_model,
                                                                                      PRE_DIVIDE_TILES_BY_255=PRE_DIVIDE_TILES_BY_255,
                                                                                      PRE_RGB_TO_BGR=PRE_RGB_TO_BGR,
                                                                                      PRE_SUBTRACT_MEAN_FROM_TILE=PRE_SUBTRACT_MEAN_FROM_TILE)
                elif current_model_mode == 'di':
                    # Create a test generator
                    region_generator = my_functions.mode_8_di_coordinates_generator(tile_dicts=region_dataset_dict,
                                                                                    batch_size=batch_size,
                                                                                    n_classes=N_CLASSES_TRAINING,
                                                                                    shuffle=False,
                                                                                    TILE_SIZE=TILE_SIZE_DIAGNOSTIC,
                                                                                    which_scale_to_use=current_scale_to_use,
                                                                                    label=current_label_to_use,
                                                                                    PRE_DIVIDE_TILES_BY_255=PRE_DIVIDE_TILES_BY_255,
                                                                                    PRE_RGB_TO_BGR=PRE_RGB_TO_BGR,
                                                                                    PRE_SUBTRACT_MEAN_FROM_TILE=PRE_SUBTRACT_MEAN_FROM_TILE)
                elif current_model_mode == 'tri':
                    # Create data generators
                    region_generator = my_functions.mode_8_tri_coordinates_generator(tile_dicts=region_dataset_dict,
                                                                                     batch_size=batch_size,
                                                                                     n_classes=N_CLASSES_TRAINING,
                                                                                     shuffle=False,
                                                                                     TILE_SIZE=TILE_SIZE_DIAGNOSTIC,
                                                                                     which_scale_to_use=current_scale_to_use,
                                                                                     label=current_label_to_use,
                                                                                     PRE_DIVIDE_TILES_BY_255=PRE_DIVIDE_TILES_BY_255,
                                                                                     PRE_RGB_TO_BGR=PRE_RGB_TO_BGR,
                                                                                     PRE_SUBTRACT_MEAN_FROM_TILE=PRE_SUBTRACT_MEAN_FROM_TILE)

                # Run model
                current_region_predictions_filename = current_wsi_summary_path + 'region_{}_predictions.pickle'.format(region_index)
                if os.path.exists(current_region_predictions_filename) and RUN_NEW_PREDICTION is False:
                    region_predictions = my_functions.pickle_load(current_region_predictions_filename)
                else:
                    region_predictions = current_deep_learning_model.predict_generator(generator=region_generator,
                                                                                       max_queue_size=MAX_QUEUE_SIZE,
                                                                                       workers=N_WORKERS,
                                                                                       use_multiprocessing=USE_MULTIPROCESSING,
                                                                                       verbose=1)
                    # Backup all data
                    my_functions.pickle_save(region_predictions, current_region_predictions_filename)

                region_class_0_probability = []
                region_class_1_probability = []
                region_class_2_probability = []
                region_class_3_probability = []

                # Loop through all tiles of current region and check prediction for each tile
                for tile_index, tile_dict in region_dataset_dict.items():
                    # Append variables to lists
                    y_true_argmax_current_wsi.append(tile_dict[current_label_to_use])
                    y_pred_argmax_current_wsi.append(np.argmax(region_predictions[tile_index], axis=0))

                    # Append prediction probabilities
                    region_class_0_probability.append(region_predictions[tile_index][0])
                    region_class_1_probability.append(region_predictions[tile_index][1])
                    if N_CLASSES_TRAINING in [3, 4]:
                        region_class_2_probability.append(region_predictions[tile_index][2])
                    if N_CLASSES_TRAINING == 4:
                        region_class_3_probability.append(region_predictions[tile_index][3])

                # Add probabilities for each class
                list_of_all_predictions_class0.extend(region_class_0_probability)
                list_of_all_predictions_class1.extend(region_class_1_probability)
                list_of_all_predictions_class2.extend(region_class_2_probability)
                list_of_all_predictions_class3.extend(region_class_3_probability)

                # break out from regions (for debugging)
                if DEBUG_MODE is True and region_index == N_REGIONS_TO_PROCESS_DEBUG_MODE - 1:
                    my_functions.my_print('DEBUG MODE: Breaking out of process loop')
                    break

            # Find WSI level prediction for each class
            # WHO04: HighGrade = 0, LowGrade = 1.
            total_sum = sum(list_of_all_predictions_class0) + sum(list_of_all_predictions_class1) + sum(list_of_all_predictions_class2) + sum(list_of_all_predictions_class3)
            wsi_probability_class0 = (sum(list_of_all_predictions_class0) / total_sum) * 100  # High Grade
            wsi_probability_class1 = (sum(list_of_all_predictions_class1) / total_sum) * 100
            wsi_probability_class2 = (sum(list_of_all_predictions_class2) / total_sum) * 100
            wsi_probability_class3 = (sum(list_of_all_predictions_class3) / total_sum) * 100

            # region FINISH
            # Calculate elapse time for current run
            elapse_time = time.time() - current_start_time
            m, s = divmod(elapse_time, 60)
            h, m = divmod(m, 60)
            model_time = '%02d:%02d:%02d' % (h, m, s)

            # Print out results
            my_functions.my_print('Finished testing model on validation set. Time: {}'.format(model_time))
            my_functions.my_print('')

            # If the first WSI, create new empty lists. If not, restore existing lists
            if not os.path.isfile(model_prediction_summary_pickle_file):
                list_of_wsi_level_ground_truth_for_all_wsi = []
                list_of_wsi_percentage_tiles_class0 = []
            else:
                # Load variables
                pickle_reader = open(model_prediction_summary_pickle_file, 'rb')
                (list_of_wsi_level_ground_truth_for_all_wsi, list_of_wsi_percentage_tiles_class0) = pickle.load(pickle_reader)
                pickle_reader.close()

            # Append values for all batches
            list_of_wsi_level_ground_truth_for_all_wsi.append(y_true_argmax_current_wsi[0])

            # Find distribution of low/high grade tiles in WSI
            wsi_percentage_tiles_class0 = (y_pred_argmax_current_wsi.count(0) / len(y_pred_argmax_current_wsi)) * 100
            list_of_wsi_percentage_tiles_class0.append(wsi_percentage_tiles_class0)

            # Save variables for classification report
            pickle_writer = open(model_prediction_summary_pickle_file, 'wb')
            pickle.dump((list_of_wsi_level_ground_truth_for_all_wsi, list_of_wsi_percentage_tiles_class0), pickle_writer)
            pickle_writer.close()

            # Write result to summary.csv file
            summary_csv = [wsi_filename_no_extension]
            summary_csv.append(NAME_OF_CLASSES_TRAINING[y_true_argmax_current_wsi[0]])  # true class
            summary_csv.append('{:.1f}%'.format(wsi_probability_class0))
            summary_csv.append('{:.1f}%'.format(wsi_probability_class1))
            if N_CLASSES_TRAINING in [3, 4]:
                summary_csv.append('{:.1f}%'.format(wsi_probability_class2))
            if N_CLASSES_TRAINING in [4]:
                summary_csv.append('{:.1f}%'.format(wsi_probability_class3))
            summary_csv.append(WHAT_MODEL_TO_LOAD)
            summary_csv.append(WHAT_MODEL_EPOCH_TO_LOAD)
            summary_csv.append(DEBUG_MODE)
            summary_csv.append(N_REGIONS_TO_PROCESS_DEBUG_MODE)
            summary_csv.append(model_time)
            summary_csv.append(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

            try:
                with open(current_model_summary_csv_file, 'a') as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(summary_csv)
            except Exception as e:
                my_functions.my_print('Error writing to file', error=True)
                my_functions.my_print(e, error=True)

            # endregion
    return current_model_name


def find_wsi_decision_thresholds(SCN_PATH, DIAGNOSTIC_TRAINING_DICTS_PATH, current_run_path, MODE_FOLDER, current_model_name,
                                 DEBUG_MODE, N_REGIONS_TO_PROCESS_DEBUG_MODE):
    wsi_list_train = os.listdir(DIAGNOSTIC_TRAINING_DICTS_PATH)
    wsi_list_train.sort()
    max_acc = 0
    list_of_wsi_percentage_tiles_class0 = []
    list_of_wsi_level_ground_truth_for_all_wsi = []

    # Loop through all WSIs
    for current_wsi_index, current_wsi_coordinate_pickle_filename in enumerate(wsi_list_train):

        wsi_filename_no_extension = current_wsi_coordinate_pickle_filename.split("list")[0][:-1]
        current_mode_summary_path = current_run_path + MODE_FOLDER
        current_model_summary_path = current_mode_summary_path + current_model_name
        current_wsi_summary_path = current_model_summary_path + wsi_filename_no_extension + '/'

        y_pred_current_wsi = []
        y_true_argmax_current_wsi = []

        # Load list containing one dict per region
        list_of_region_dicts_in_current_wsi = my_functions.pickle_load(DIAGNOSTIC_TRAINING_DICTS_PATH + current_wsi_coordinate_pickle_filename)

        # Process each region in current WSI
        for region_index, current_region in enumerate(list_of_region_dicts_in_current_wsi):

            current_region_predictions_filename = current_wsi_summary_path + 'region_{}_predictions.pickle'.format(region_index)
            region_predictions = my_functions.pickle_load(current_region_predictions_filename)

            # Reset dataset dict
            region_dataset_dict = dict()

            # Extract each tile
            for _, value in current_region['tile_dict'].items():
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

                # Add tile to dict
                tempvalue = value.copy()
                tempvalue['augmentarg'] = 1
                tempvalue['path'] = SCN_PATH + current_region['wsi_filename'].split(".")[0] + '/' + current_region['wsi_filename']
                tempvalue['WHO73'] = current_region['WHO73']
                tempvalue['Recurrence'] = current_region['Recurrence']
                tempvalue['Stage'] = current_region['Stage']
                tempvalue['Progression'] = current_region['Progression']
                tempvalue['WHO04'] = current_region['WHO04']
                region_dataset_dict[len(region_dataset_dict)] = tempvalue

            # Loop through all tiles of current region and check prediction for each tile
            for tile_index, tile_dict in region_dataset_dict.items():
                # Append variables to lists
                y_true_argmax_current_wsi.append(tile_dict['WHO04'])
                y_pred_current_wsi.append(np.argmax(region_predictions[tile_index], axis=0))

            # break out from regions (for debugging)
            if DEBUG_MODE is True and region_index == N_REGIONS_TO_PROCESS_DEBUG_MODE - 1:
                my_functions.my_print('DEBUG MODE: Breaking out of process loop')
                break

        # Find distribution of low/high grade tiles in WSI
        wsi_percentage_tiles_class0 = (y_pred_current_wsi.count(0) / len(y_pred_current_wsi)) * 100

        list_of_wsi_percentage_tiles_class0.append(round(wsi_percentage_tiles_class0, 2))
        list_of_wsi_level_ground_truth_for_all_wsi.append(y_true_argmax_current_wsi[0])

    # Loop through all wsi thresholds
    n = 0
    for wsi_threshold_candidate in range(0, 49):
        n += 1
        y_pred = []
        for class0_percentage_in_current_wsi in list_of_wsi_percentage_tiles_class0:
            if class0_percentage_in_current_wsi > wsi_threshold_candidate:
                # value is over threshold -> high_grade (0)
                y_pred.append(0)
            else:
                y_pred.append(1)

        current_wsi_threshold_candidate_accuracy = accuracy_score(list_of_wsi_level_ground_truth_for_all_wsi, y_pred)
        n_correct_classified = accuracy_score(list_of_wsi_level_ground_truth_for_all_wsi, y_pred, normalize=False)

        # If new record, reset best tile threshold list
        if current_wsi_threshold_candidate_accuracy > max_acc:
            max_acc = current_wsi_threshold_candidate_accuracy
            n_correct_classified_best = n_correct_classified
            list_of_best_wsi_level_thresholds = []

        # Append to list
        if current_wsi_threshold_candidate_accuracy >= max_acc:
            list_of_best_wsi_level_thresholds.append(wsi_threshold_candidate)

    decision_threshold_wsi = list_of_best_wsi_level_thresholds[int(len(list_of_best_wsi_level_thresholds) // 2)]

    print('')
    print('Overall best thresholds:')
    print('Out of {} candidate thresholds, We found {} decision thresholds which maximized the accuracy'.format(n, len(list_of_best_wsi_level_thresholds)))
    print('Max acc: {}, with {}/{} correctly predicted WSIs.'.format(round(max_acc, 2), n_correct_classified_best, len(wsi_list_train)))
    print('Best wsi decision threshold: ', decision_threshold_wsi)

    # Write results to textfile
    with open(current_model_summary_path + "decision_threshold.txt", "w") as text_file:
        text_file.write('Overall best thresholds: \n')
        text_file.write('Out of {} candidate thresholds, We found {} decision thresholds which maximized the accuracy \n'.format(n, len(list_of_best_wsi_level_thresholds)))
        text_file.write('Max acc: {}, with {}/{} correctly predicted WSIs. \n'.format(round(max_acc, 2), n_correct_classified_best, len(wsi_list_train)))
        text_file.write('Best wsi decision threshold: {}'.format(decision_threshold_wsi))

    # Save decision threshold
    my_functions.pickle_save(decision_threshold_wsi, current_model_summary_path + 'decision_threshold_wsi.obj')
