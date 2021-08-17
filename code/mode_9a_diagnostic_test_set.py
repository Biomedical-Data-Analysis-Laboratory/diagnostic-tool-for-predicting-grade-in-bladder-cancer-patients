from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adadelta
from keras.optimizers import Adam
from keras.optimizers import Adagrad
from keras.optimizers import Adamax
from keras.optimizers import Nadam
from keras.optimizers import SGD
import my_functions
import my_constants
import numpy as np
import matplotlib
import operator
import datetime
import pyvips
import pickle
import keras
import time
import csv
import sys
import cv2
import os


def diagnostic_test_set(current_run_path, METADATA_FOLDER, MODEL_WEIGHT_FOLDER, N_CHANNELS, WHAT_MODEL_TO_LOAD,
                        WHAT_MODEL_EPOCH_TO_LOAD, MODE_FOLDER, DIAGNOSTIC_TEST_DICTS_PATH, SCN_PATH,
                        MODELS_AND_LOSS_ARRAY_DTL_PICKLE_FILE, ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE,
                        TRAINING_DATA_DTL_PICKLE_FILE, SUMMARY_CSV_FILE_PATH, tf_version, TILE_SIZE_DIAGNOSTIC,
                        current_wsi_coordinate_pickle_filename, wsi_filename_no_extension, wsi_dataset_folder,
                        wsi_dataset_file_path, TILES_TO_SHOW, DEBUG_MODE, MAX_QUEUE_SIZE, N_WORKERS, USE_MULTIPROCESSING,
                        N_REGIONS_TO_PROCESS_DEBUG_MODE, PRE_DIVIDE_TILES_BY_255, PRE_RGB_TO_BGR,
                        PRE_SUBTRACT_MEAN_FROM_TILE, RUN_NEW_PREDICTION, current_wsi_index, MODE_FDT_FOLDER):
    # region FILE INIT

    # Start timer
    current_start_time = time.time()

    # Large batch size always gives error message on test-set. Manually reduce batch size for testing.
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

            my_functions.my_print('Testing model on diagnostic test set - {}, Freeze:{}, Layer_{}, optimizer:{}, learning rate:{}, '
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

            # Make folder for current WSI
            current_wsi_summary_path = current_model_summary_path + wsi_filename_no_extension + '/'
            os.makedirs(current_wsi_summary_path, exist_ok=True)

            # Define summary files
            model_prediction_summary_pickle_file = current_model_summary_path + 'model_prediction_summary.obj'
            current_model_summary_csv_file = current_model_summary_path + 'Test_summary.csv'

            # Load decision threshold if exist
            decision_threshold_wsi_pickle_file = current_run_path + MODE_FDT_FOLDER + current_model_name + 'decision_threshold_wsi.obj'
            decision_threshold_wsi = my_functions.pickle_load(decision_threshold_wsi_pickle_file)

            # Delete summary files if they exist. But only on start, so check that index is zero.
            if current_wsi_index == 0:
                if os.path.exists(model_prediction_summary_pickle_file):
                    os.remove(model_prediction_summary_pickle_file)
                if os.path.exists(current_model_summary_csv_file):
                    os.remove(current_model_summary_csv_file)

            # Make an array for top row in summary CSV file
            csv_array = ['IMAGE_NAME', 'TRUE', 'PRED']
            for name_of_class in NAME_OF_CLASSES_TRAINING:
                csv_array.append(name_of_class + '_proportion')
            for name_of_class in NAME_OF_CLASSES_TRAINING:
                csv_array.append(name_of_class + '_tiles')
            csv_array.append('Decision_threshold_wsi')
            csv_array.append('WHAT_MODEL_TO_LOAD')
            csv_array.append('WHAT_MODEL_EPOCH_TO_LOAD')
            csv_array.append('TILES_TO_SHOW')
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
                current_deep_learning_model, \
                current_latent_size = my_functions.get_mono_scale_model(img_width=TILE_SIZE_DIAGNOSTIC,
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
                current_deep_learning_model, \
                current_latent_size = my_functions.get_di_scale_model(img_width=TILE_SIZE_DIAGNOSTIC,
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
                current_deep_learning_model, \
                current_latent_size = my_functions.get_tri_scale_model(img_width=TILE_SIZE_DIAGNOSTIC,
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
                     current_best_val_loss_data, current_best_val_acc_data, current_best_val_acc_epoch_data, current_best_val_loss_epoch_data,
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

            # region OVERVIEW IMAGE INIT
            # Save overview image
            current_wsi_overview_image_filename = '{}/{}#overview.jpeg'.format(current_wsi_summary_path, wsi_filename_no_extension)
            image_25x = pyvips.Image.new_from_file(wsi_dataset_file_path, level=2).flatten().rot(1)
            offset_x_x25, offset_y_x25, width_25x, height_25x = my_functions.remove_white_background_v3(input_img=image_25x, PADDING=0, folder_path=wsi_dataset_folder)
            overview_img = image_25x.extract_area(offset_x_x25, offset_y_x25, width_25x, height_25x)
            overview_img.jpegsave(current_wsi_overview_image_filename, Q=100)

            # Set normalization limits for colormap
            font_scale = width_25x / 3000
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

            # Read overview image using cv2. We need to read it many times to create different images
            overview_jpeg_file = cv2.imread(current_wsi_overview_image_filename, cv2.IMREAD_UNCHANGED)
            predicted_argmax_image = cv2.imread(current_wsi_overview_image_filename, cv2.IMREAD_UNCHANGED)
            overview_jpeg_file_heatmap = dict()
            for i in range(N_CLASSES_TRAINING):
                overview_jpeg_file_heatmap[i] = cv2.imread(current_wsi_overview_image_filename, cv2.IMREAD_UNCHANGED)

            # Add colorbar to heatmap
            box_size = int(width_25x * 0.02)
            for i in range(N_CLASSES_TRAINING):
                for n in range(0, 12):
                    # Get RGBA value for current prediction value
                    rgba_value2 = matplotlib.cm.jet(norm(n / 10), bytes=False)
                    # Remove alpha channel RGBA -> RGB
                    rgb_value = rgba_value2[0:3]
                    # Convert from RGB -> BGR. Also, convert from 0-1 -> 0-255 values.
                    bgr_value = (rgb_value[2] * 255, rgb_value[1] * 255, rgb_value[0] * 255)

                    # Set start and end coordinates
                    x_start_25x = width_25x - 3 * box_size
                    x_end_25x = x_start_25x + box_size
                    y_start_25x = 11 * box_size - box_size * n
                    y_end_25x = y_start_25x + box_size
                    text_coord_x = x_end_25x + 20
                    text_coord_y = y_start_25x + box_size // 2

                    # Draw rectangles and text
                    if n < 11:
                        cv2.rectangle(overview_jpeg_file_heatmap[i], (x_start_25x, y_start_25x), (x_end_25x, y_end_25x),
                                      bgr_value, -1)
                        cv2.putText(img=overview_jpeg_file_heatmap[i],
                                    text='{}%'.format(n * 10),
                                    org=(text_coord_x, text_coord_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=font_scale, color=(0, 0, 0), thickness=3)
                    else:
                        cv2.putText(img=overview_jpeg_file_heatmap[i],
                                    text='{}'.format(NAME_OF_CLASSES_TRAINING_DISPLAYNAME[i]),
                                    org=(text_coord_x - 200, text_coord_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=font_scale, color=(0, 0, 0), thickness=3)
            # endregion

            # init
            y_true_argmax_current_wsi = []
            y_pred_argmax_current_wsi = []
            list_of_all_predictions_class0 = []
            list_of_all_predictions_class1 = []
            list_of_all_predictions_class2 = []
            list_of_all_predictions_class3 = []

            # Load list containing one dict per region
            list_of_region_dicts_in_current_wsi = my_functions.pickle_load(DIAGNOSTIC_TEST_DICTS_PATH + current_wsi_coordinate_pickle_filename)
            """
            Example of list_of_region_dicts_in_current_wsi:
                [   'tissue_type': 'urothelium', 
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

                region_dataset_size = len(region_dataset_dict)

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
                current_region_predictions_path = current_model_summary_path + 'pickle_data/' + wsi_filename_no_extension + '/'
                os.makedirs(current_region_predictions_path, exist_ok=True)
                current_region_predictions_filename = current_region_predictions_path + 'region_{}_predictions.pickle'.format(region_index)

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

                # Create lists
                region_class_0_probability = []
                region_class_1_probability = []
                region_class_2_probability = []
                region_class_3_probability = []

                # Loop through all tiles of current region and check prediction for each tile
                for tile_index, tile_dict in region_dataset_dict.items():

                    # Get y_true and y_pred for the current tile
                    tile_y_true = tile_dict[current_label_to_use]
                    tile_y_pred = np.argmax(region_predictions[tile_index], axis=0)

                    # Append variables to lists
                    y_true_argmax_current_wsi.append(tile_y_true)
                    y_pred_argmax_current_wsi.append(tile_y_pred)

                    # Append prediction probabilities
                    region_class_0_probability.append(region_predictions[tile_index][0])
                    region_class_1_probability.append(region_predictions[tile_index][1])
                    if N_CLASSES_TRAINING in [3, 4]:
                        region_class_2_probability.append(region_predictions[tile_index][2])
                    if N_CLASSES_TRAINING == 4:
                        region_class_3_probability.append(region_predictions[tile_index][3])

                    # Draw tiles on the overview image, start by finding start and end coordinates
                    x_start_25x = tile_dict['coordinates_25x'][0] - offset_x_x25
                    y_start_25x = tile_dict['coordinates_25x'][1] - offset_y_x25
                    x_start_100x = int((tile_dict['coordinates_25x'][0] - offset_x_x25 + (TILE_SIZE_DIAGNOSTIC / 2)) - (TILE_SIZE_DIAGNOSTIC * my_constants.Scale_between_25x_100x) / 2)
                    y_start_100x = int((tile_dict['coordinates_25x'][1] - offset_y_x25 + (TILE_SIZE_DIAGNOSTIC / 2)) - (TILE_SIZE_DIAGNOSTIC * my_constants.Scale_between_25x_100x) / 2)
                    x_start_400x = int((tile_dict['coordinates_25x'][0] - offset_x_x25 + (TILE_SIZE_DIAGNOSTIC / 2)) - (TILE_SIZE_DIAGNOSTIC * my_constants.Scale_between_25x_400x) / 2)
                    y_start_400x = int((tile_dict['coordinates_25x'][1] - offset_y_x25 + (TILE_SIZE_DIAGNOSTIC / 2)) - (TILE_SIZE_DIAGNOSTIC * my_constants.Scale_between_25x_400x) / 2)

                    x_end_25x = x_start_25x + TILE_SIZE_DIAGNOSTIC
                    y_end_25x = y_start_25x + TILE_SIZE_DIAGNOSTIC
                    x_end_100x = int(x_start_100x + TILE_SIZE_DIAGNOSTIC * my_constants.Scale_between_25x_100x)
                    y_end_100x = int(y_start_100x + TILE_SIZE_DIAGNOSTIC * my_constants.Scale_between_25x_100x)
                    x_end_400x = int(x_start_400x + TILE_SIZE_DIAGNOSTIC * my_constants.Scale_between_25x_400x)
                    y_end_400x = int(y_start_400x + TILE_SIZE_DIAGNOSTIC * my_constants.Scale_between_25x_400x)

                    # Draw tiles
                    if tile_y_true == tile_y_pred:
                        # Correct prediction, draw a green tile (BGR = Blue, Green, Red)
                        if '25x' in TILES_TO_SHOW:
                            cv2.rectangle(overview_jpeg_file, (x_start_25x, y_start_25x), (x_end_25x, y_end_25x), (0, 255, 0), 3)
                        if '100x' in TILES_TO_SHOW:
                            cv2.rectangle(overview_jpeg_file, (x_start_100x, y_start_100x), (x_end_100x, y_end_100x), (0, 255, 0), 3)
                        if '400x' in TILES_TO_SHOW:
                            cv2.rectangle(overview_jpeg_file, (x_start_400x, y_start_400x), (x_end_400x, y_end_400x), (0, 255, 0), 3)
                    elif (tile_y_true != tile_y_pred) and (tile_y_pred != -1):
                        # Wrong prediction, draw a red tile
                        if '25x' in TILES_TO_SHOW:
                            cv2.rectangle(overview_jpeg_file, (x_start_25x, y_start_25x), (x_end_25x, y_end_25x), (0, 0, 255), 3)
                        if '100x' in TILES_TO_SHOW:
                            cv2.rectangle(overview_jpeg_file, (x_start_100x, y_start_100x), (x_end_100x, y_end_100x), (0, 0, 255), 3)
                        if '400x' in TILES_TO_SHOW:
                            cv2.rectangle(overview_jpeg_file, (x_start_400x, y_start_400x), (x_end_400x, y_end_400x), (0, 0, 255), 3)

                    # Draw tiles on heatmap
                    for i in range(N_CLASSES_TRAINING):
                        # Get RGBA value for current prediction value
                        rgba_value2 = matplotlib.cm.jet(norm(region_predictions[tile_index][i]), bytes=False)
                        # Remove alpha channel RGBA -> RGB
                        rgb_value = rgba_value2[0:3]
                        # Convert from RGB -> BGR. Also, convert from 0-1 -> 0-255 values.
                        bgr_value = (rgb_value[2] * 255, rgb_value[1] * 255, rgb_value[0] * 255)

                        # Draw rectangles
                        if '25x' in TILES_TO_SHOW:
                            cv2.rectangle(overview_jpeg_file_heatmap[i], (x_start_25x, y_start_25x), (x_end_25x, y_end_25x), bgr_value, -1)
                        if '100x' in TILES_TO_SHOW:
                            cv2.rectangle(overview_jpeg_file_heatmap[i], (x_start_100x, y_start_100x), (x_end_100x, y_end_100x), bgr_value, -1)
                        if '400x' in TILES_TO_SHOW:
                            cv2.rectangle(overview_jpeg_file_heatmap[i], (x_start_400x, y_start_400x), (x_end_400x, y_end_400x), bgr_value, -1)

                        # Draw solid color tiles on argmax_predicte_image, one color for each predicted class
                        # high_grade (RGB) = (128, 0, 0)
                        # low_grade (RGB) = (0, 130, 200)
                        if tile_y_pred == 0:
                            cv2.rectangle(predicted_argmax_image, (x_start_400x, y_start_400x), (x_end_400x, y_end_400x), (0, 0, 128), -1)
                        else:
                            cv2.rectangle(predicted_argmax_image, (x_start_400x, y_start_400x), (x_end_400x, y_end_400x), (200, 130, 0), -1)

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
            wsi_probability_class0 = (sum(list_of_all_predictions_class0) / total_sum) * 100
            wsi_probability_class1 = (sum(list_of_all_predictions_class1) / total_sum) * 100
            wsi_probability_class2 = (sum(list_of_all_predictions_class2) / total_sum) * 100
            wsi_probability_class3 = (sum(list_of_all_predictions_class3) / total_sum) * 100

            # Find final prediction by counting all predicted tiles, and compute the distribution of these.
            if current_label_to_use == 'WHO04':
                wsi_percentage_tiles_class0 = (y_pred_argmax_current_wsi.count(0) / len(y_pred_argmax_current_wsi)) * 100
                wsi_percentage_tiles_class1 = (y_pred_argmax_current_wsi.count(1) / len(y_pred_argmax_current_wsi)) * 100

                # If there are more high_grade tissue in current WSI than the decision_threshold, final prediction is high_grade. Else, final prediction is low_grade
                if wsi_percentage_tiles_class0 >= decision_threshold_wsi:
                    final_prediction_index = 0
                else:
                    final_prediction_index = 1

            else:
                # If other label than WHO04 is used, major class is final prediction.
                final_prediction_index = np.argmax([wsi_probability_class0, wsi_probability_class1, wsi_probability_class2, wsi_probability_class3])

            # Put in prediction summary text in bottom of overview image
            for i in range(N_CLASSES_TRAINING):
                # Black box behind text (bottom left)
                if N_CLASSES_TRAINING == 2:
                    cv2.rectangle(overview_jpeg_file_heatmap[i],
                                  (30, overview_img.height - 450),
                                  (1400, overview_img.height - 100),
                                  (0, 0, 0), cv2.FILLED)
                elif N_CLASSES_TRAINING == 3:
                    cv2.rectangle(overview_jpeg_file_heatmap[i],
                                  (30, overview_img.height - 450),
                                  (1400, overview_img.height - 80),
                                  (0, 0, 0), cv2.FILLED)
                elif N_CLASSES_TRAINING == 4:
                    cv2.rectangle(overview_jpeg_file_heatmap[i],
                                  (30, overview_img.height - 450),
                                  (1400, overview_img.height - 20),
                                  (0, 0, 0), cv2.FILLED)

                # 'Final prediction based on regions: Grade 3'
                cv2.putText(img=overview_jpeg_file_heatmap[i],
                            text='Predicted class: {}'.format(NAME_OF_CLASSES_TRAINING_DISPLAYNAME[final_prediction_index]),
                            org=(40, overview_img.height - 390),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 255), thickness=3)
                # 'True class: grade 2'
                cv2.putText(img=overview_jpeg_file_heatmap[i],
                            text='True class: {}'.format(name_and_index_of_classes[list_of_region_dicts_in_current_wsi[0][current_label_to_use]]['display_name']),
                            org=(40, overview_img.height - 320),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 255), thickness=3)
                # 'Grade 1: 60.3%'
                cv2.putText(img=overview_jpeg_file_heatmap[i],
                            text='{} coverage: {:.1f}%'.format(NAME_OF_CLASSES_TRAINING_DISPLAYNAME[0], wsi_percentage_tiles_class0),
                            org=(40, overview_img.height - 250),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 255), thickness=3)
                # 'Grade 2: 60.3%'
                cv2.putText(img=overview_jpeg_file_heatmap[i],
                            text='{} coverage: {:.1f}%'.format(NAME_OF_CLASSES_TRAINING_DISPLAYNAME[1], wsi_percentage_tiles_class1),
                            org=(40, overview_img.height - 180),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 255), thickness=3)
                if N_CLASSES_TRAINING in [3, 4]:
                    # 'Grade 3: 60.3%'
                    cv2.putText(img=overview_jpeg_file_heatmap[i],
                                text='{} average prob.: {:.1f}%'.format(NAME_OF_CLASSES_TRAINING_DISPLAYNAME[2], wsi_probability_class2),
                                org=(40, overview_img.height - 110),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 255), thickness=3)
                if N_CLASSES_TRAINING == 4:
                    # 'Class 4: 60.3%'
                    cv2.putText(img=overview_jpeg_file_heatmap[i],
                                text='{} average prob.: {:.1f}%'.format(NAME_OF_CLASSES_TRAINING_DISPLAYNAME[3], wsi_probability_class3),
                                org=(40, overview_img.height - 40),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 255), thickness=3)

            # Black box behind text (bottom left)
            if N_CLASSES_TRAINING == 2:
                cv2.rectangle(overview_jpeg_file,
                              (30, overview_img.height - 450),
                              (1400, overview_img.height - 100),
                              (0, 0, 0), cv2.FILLED)
            elif N_CLASSES_TRAINING == 3:
                cv2.rectangle(overview_jpeg_file,
                              (30, overview_img.height - 450),
                              (1400, overview_img.height - 80),
                              (0, 0, 0), cv2.FILLED)
            elif N_CLASSES_TRAINING == 4:
                cv2.rectangle(overview_jpeg_file,
                              (30, overview_img.height - 450),
                              (1400, overview_img.height - 20),
                              (0, 0, 0), cv2.FILLED)
            # 'Final prediction based on regions: Grade 3'
            cv2.putText(img=overview_jpeg_file,
                        text='Predicted class: {}'.format(NAME_OF_CLASSES_TRAINING_DISPLAYNAME[final_prediction_index]),
                        org=(40, overview_img.height - 390),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 255), thickness=3)
            # 'True class: grade 2'
            cv2.putText(img=overview_jpeg_file,
                        text='True class: {}'.format(name_and_index_of_classes[list_of_region_dicts_in_current_wsi[0][current_label_to_use]]['display_name']),
                        org=(40, overview_img.height - 320),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 255), thickness=3)

            # Save overview image
            current_wsi_overview_image_w_tiles_filename = '{}/{}#tiles.jpeg'.format(current_wsi_summary_path, wsi_filename_no_extension)
            cv2.imwrite(current_wsi_overview_image_w_tiles_filename, overview_jpeg_file, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            for i in range(N_CLASSES_TRAINING):
                current_wsi_overview_image_heatmap_filename = '{}/{}#heatmap_{}.jpeg'.format(current_wsi_summary_path, wsi_filename_no_extension, NAME_OF_CLASSES_TRAINING[i])
                cv2.imwrite(current_wsi_overview_image_heatmap_filename, overview_jpeg_file_heatmap[i], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            predicted_argmax_image_filename = '{}/{}#argmax_pred.jpeg'.format(current_wsi_summary_path, wsi_filename_no_extension)
            cv2.imwrite(predicted_argmax_image_filename, predicted_argmax_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            # region FINISH
            # Calculate elapse time for current run
            elapse_time = time.time() - current_start_time
            m, s = divmod(elapse_time, 60)
            h, m = divmod(m, 60)
            model_time = '%02d:%02d:%02d' % (h, m, s)

            # Print out results
            my_functions.my_print('Finished testing model on test set. Time: {}'.format(model_time))
            my_functions.my_print('')

            # If the first WSI, create new empty lists. If not, restore existing lists
            if not os.path.isfile(model_prediction_summary_pickle_file):
                list_of_tile_level_argmax_predictions_for_all_wsi = []
                list_of_wsi_level_ground_truth_for_all_wsi = []
                list_of_wsi_level_prediction_for_all_wsi = []
                cm_label = list(range(N_CLASSES_TRAINING))
            else:
                # Load variables
                pickle_reader = open(model_prediction_summary_pickle_file, 'rb')
                (list_of_tile_level_argmax_predictions_for_all_wsi, list_of_wsi_level_ground_truth_for_all_wsi, list_of_wsi_level_prediction_for_all_wsi, cm_label) = pickle.load(pickle_reader)
                pickle_reader.close()

            # Append values for all batches
            list_of_tile_level_argmax_predictions_for_all_wsi.extend(y_pred_argmax_current_wsi)
            list_of_wsi_level_ground_truth_for_all_wsi.append(y_true_argmax_current_wsi[0])
            list_of_wsi_level_prediction_for_all_wsi.append(final_prediction_index)

            # Save variables for classification report
            pickle_writer = open(model_prediction_summary_pickle_file, 'wb')
            pickle.dump((list_of_tile_level_argmax_predictions_for_all_wsi, list_of_wsi_level_ground_truth_for_all_wsi, list_of_wsi_level_prediction_for_all_wsi, cm_label), pickle_writer)
            pickle_writer.close()

            # Write result to summary.csv file
            summary_csv = [wsi_filename_no_extension]
            summary_csv.append(NAME_OF_CLASSES_TRAINING[y_true_argmax_current_wsi[0]])  # true class
            summary_csv.append(NAME_OF_CLASSES_TRAINING[final_prediction_index])  # predicted class
            summary_csv.append('{:.1f}%'.format(wsi_percentage_tiles_class0))
            summary_csv.append('{:.1f}%'.format(wsi_percentage_tiles_class1))
            if N_CLASSES_TRAINING in [3, 4]:
                summary_csv.append('{:.1f}%'.format(wsi_probability_class2))
            if N_CLASSES_TRAINING in [4]:
                summary_csv.append('{:.1f}%'.format(wsi_probability_class3))
            summary_csv.append(y_pred_argmax_current_wsi.count(0))
            summary_csv.append(y_pred_argmax_current_wsi.count(1))
            summary_csv.append(round(decision_threshold_wsi, 2))
            summary_csv.append(WHAT_MODEL_TO_LOAD)
            summary_csv.append(WHAT_MODEL_EPOCH_TO_LOAD)
            summary_csv.append(TILES_TO_SHOW)
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

            # Write result to summary.csv file
            my_functions.summary_csv_file_update(SUMMARY_CSV_FILE_PATH=SUMMARY_CSV_FILE_PATH,
                                                 MODE='9a',
                                                 model_name=current_model_name,
                                                 label=current_label_to_use,
                                                 base_model=current_base_model,
                                                 freeze_base_model=current_freeze_base_model,
                                                 blocks_to_unfreeze_vgg16_vgg19='N/A',
                                                 delayed_unfreeze_start_epoche_vgg16_vgg19='N/A',
                                                 base_model_pooling=current_base_model_pooling,
                                                 training_samples='N/A',
                                                 validation_samples='N/A',
                                                 test_samples=region_dataset_size,
                                                 layer_config=current_layer_config,
                                                 augment_classes='N/A',
                                                 augment_multiplier='N/A',
                                                 learning_rate=current_learning_rate,
                                                 batch_size=batch_size,
                                                 n_neurons1=current_n_neurons1,
                                                 n_neurons2=current_n_neurons2,
                                                 n_neurons3=current_n_neurons3,
                                                 EARLY_STOPPING_PATIENCE='N/A',
                                                 dropout=current_dropout,
                                                 best_train_loss='N/A',
                                                 best_train_acc='N/A',
                                                 best_val_loss='N/A',
                                                 best_val_acc='N/A',
                                                 best_val_loss_epoch='N/A',
                                                 best_val_acc_epoch='N/A',
                                                 trained_epoches='N/A',
                                                 total_epochs='N/A',
                                                 latent_size=current_latent_size,
                                                 compression='N/A',
                                                 model_time=model_time,
                                                 optimizer=current_optimizer,
                                                 ReduceLRstatus='N/A',
                                                 n_trainable_parameters_start='N/A',
                                                 n_non_trainable_parameters_start='N/A',
                                                 n_trainable_parameters_end='N/A',
                                                 n_non_trainable_parameters_end='N/A',
                                                 python_version=sys.version.split(" ")[0],
                                                 keras_version=keras.__version__,
                                                 tf_version=tf_version,
                                                 tile_size=TILE_SIZE_DIAGNOSTIC)

            # endregion
    return current_model_name, N_CLASSES_TRAINING, NAME_OF_CLASSES_TRAINING_DISPLAYNAME, NAME_OF_CLASSES_TRAINING


def diagnostic_test_final_score(current_run_path, MODE_FOLDER, current_model_name, N_CLASSES_TRAINING,
                                NAME_OF_CLASSES_TRAINING_DISPLAYNAME, NAME_OF_CLASSES_TRAINING):
    my_functions.my_print('Final predictions')
    my_functions.my_print('')
    my_functions.my_print('')

    # Define current_mode_summary_path
    current_mode_summary_path = current_run_path + MODE_FOLDER
    current_model_summary_path = current_mode_summary_path + current_model_name

    # Load variables for classification report
    pickle_reader = open(current_model_summary_path + 'model_prediction_summary.obj', 'rb')
    (list_of_tile_level_argmax_predictions_for_all_wsi, list_of_wsi_level_ground_truth_for_all_wsi, list_of_wsi_level_prediction_for_all_wsi, cm_label) = pickle.load(pickle_reader)
    pickle_reader.close()


    # region RESULT BASED ON WSIs
    # Confusion Matrix
    cm = confusion_matrix(y_true=list_of_wsi_level_ground_truth_for_all_wsi,
                          y_pred=list_of_wsi_level_prediction_for_all_wsi,
                          labels=cm_label,
                          sample_weight=None)

    # Calculate total True Positive (TP), False Negative (FN) and False Positive (FP)
    TP_tot = 0
    FN_tot = 0
    FP_tot = 0
    for row in range(N_CLASSES_TRAINING):
        for col in range(N_CLASSES_TRAINING):
            if row == col:
                TP_tot += cm[row][col]
            else:
                FN_tot += cm[row][col]
                FP_tot += cm[row][col]

    # Calculate micro-average F1-Score
    F1_tot = (2 * TP_tot) / (2 * TP_tot + FN_tot + FP_tot)
    my_functions.my_print('Test dataset micro-average F1-Score: {:.2f} (WSI-level)'.format(F1_tot))
    my_functions.my_print('')

    # Define a title
    cm_title = 'Test set - WSI'

    # Save confusion matrix
    my_functions.plot_confusion_matrix(cm=cm,
                                       epoch='model_wsi',
                                       classes=NAME_OF_CLASSES_TRAINING_DISPLAYNAME,
                                       SUMMARY_PATH=current_model_summary_path,
                                       folder_name='Confusion_matrix_test',
                                       title=cm_title)

    cm = np.round(cm / cm.sum(axis=1, keepdims=True), 3)

    # Save confusion matrix (normalized)
    my_functions.plot_confusion_matrix(cm=cm,
                                       epoch='model_wsi_normalized',
                                       classes=NAME_OF_CLASSES_TRAINING_DISPLAYNAME,
                                       SUMMARY_PATH=current_model_summary_path,
                                       folder_name='Confusion_matrix_test',
                                       title=cm_title)

    # Compute classification report
    classification_report_total = classification_report(y_true=list_of_wsi_level_ground_truth_for_all_wsi,
                                                        y_pred=list_of_wsi_level_prediction_for_all_wsi,
                                                        labels=cm_label,
                                                        target_names=NAME_OF_CLASSES_TRAINING,
                                                        sample_weight=None,
                                                        digits=8)

    # Parse the classification report, so we can save it to a CSV file
    tmp = list()
    for row in classification_report_total.split("\n"):
        parsed_row = [x for x in row.split(" ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)

    # Add an empty item to line up header in CSV file
    tmp[0].insert(0, '')

    # Save classification report to CSV
    with open(current_mode_summary_path + current_model_name + 'Test_classification_report_all_wsi_wsi.csv', 'w') as newFile:
        newFileWriter = csv.writer(newFile, delimiter=';', lineterminator='\r', quoting=csv.QUOTE_MINIMAL)
        for rows in range(len(tmp)):
            newFileWriter.writerow(tmp[rows])

    my_functions.my_print('Mode 9a finished.')
    # endregion
