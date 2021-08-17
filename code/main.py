import os
# Select which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# region IMPORTS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = True
import mode_8a_diagnostic_learning
import mode_8c_find_decision_threshold
import mode_9a_diagnostic_test_set
import my_functions
import time
import sys

# endregion


if __name__ == '__main__':

    # region MODE 8a - DIAGNOSTIC LEARNING - MONO/DI/TRI - TRAIN/VAL - DTL
    TRAIN_DIAGNOSTIC_MODEL_DTL = True
    ######## HYPERPARAMETERS ###########################
    EPOCHES_DTL                                   = 1
    base_model_DTL                                = ['VGG16']  # Base model. Supported models: VGG16 and VGG19
    base_model_pooling_DTL                        = ['avg']  # Optional pooling mode for feature extraction. Must be one of: 'None', 'avg' or 'max'
    layer_config_DTL                              = ['config2_drop']  # ['config0', 'config1', 'config2', 'config3', 'config1_drop', 'config2_drop', 'config3_drop']
    which_model_mode_to_use_DTL                   = ['mono']  # Available: mono, di and tri.
    which_scale_to_use_mono_DTL                   = ['400x']  # Which magnification scale to load images from. Can be [400x, 100x, 25x]
    which_scale_to_use_di_DTL                     = [['100x', '400x']]  # Which magnification scale to load images from. Can be [['100x', '400x'], ['25x', '400x'], ['25x', '100x']]
    WHAT_LABELS_TO_USE_DTL                        = ['WHO04']  # Available: Recurrence, Progression, Stage, WHO04, WHO73.
    learning_rate_DTL                             = [0.001]  # Comma separated list, one model will be trained for each value.
    n_neurons_first_layer_DTL                     = [4096]  # Comma separated list, one model will be trained for each value.
    n_neurons_second_layer_DTL                    = [4096]  # Comma separated list, one model will be trained for each value.
    n_neurons_third_layer_DTL                     = [0]  # Comma separated list, one model will be trained for each value.
    dropout_DTL                                   = [0.5]  # Comma separated list, one model will be trained for each value. float between 0-0.99. Setting to 0 or 1 will ignore the dropout layer.
    EARLY_STOP_PATIENCE_DTL                       = 5  # number of epochs with no improvement after which training will be stopped. Set to a value larger than EPOCHES ti disable.
    REDUCE_LR_PATIENCE_DTL                        = 50  # If validation loss does not decrease for this many epoches, lower the learning rate. Set to a value larger than EPOCHES ti disable.
    ######## FREEZE / UNFREEZING #########################
    freeze_base_model_DTL                         = [True]  # True=Freeze base model. False=Unfreeze base model. Hybrid=Freeze base model, but then unfreeze, use parameters below.
    blocks_to_unfreeze_vgg16_vgg19_DTL            = ['block3', 'block4', 'block5']  # Which blocks to unfreeze. ['block1', 'block2', 'block3', 'block4', 'block5']. Only for Hybrid.
    delayed_unfreeze_start_epoche_vgg16_vgg19_DTL = 5  # At which epoche should the model unfreeze the blocks above. Only for Hybrid.
    ######## SETTINGS ####################################
    SAVE_CONF_MAT_AND_REPORT_ON_END_DTL           = False
    ######## DATASET #####################################
    SMALL_DATASET_DEBUG_MODE_DTL                  = True  # True=Load a small fraction of the dataset for debugging only. False=train on full dataset.
    BALANCE_DATASET_BY_AUGMENTATION_DTL           = False  # Augment tiles to balance the dataset. Only for Training dataset.
    AUGMENT_MINORITY_WSI_DTL                      = False  # Augment tiles in WSIs with less tiles than DESIRED_TILES_IN_EACH_WSI_DTL
    DESIRED_TILES_IN_EACH_WSI_DTL                 = 500  # How many tiles should be in each WSI.
    ######## PREPROCESSING #####################################
    PRE_DIVIDE_TILES_BY_255_DTL                   = True  # Preprocessing. Divide all pixel values by 255 to normalise values from 0-255 into 0-1.
    PRE_RGB_TO_BGR_DTL                            = True  # Preprocessing. Switch the color channels from RGB to BGR. (VGG16 pretrained weights are trained using BGR.)
    PRE_SUBTRACT_MEAN_FROM_TILE_DTL               = False  # Preprocessing. Subtract mean value from tiles. Mean values are stored in def dataset_preprocessing().
    # endregion

    # region MODE 8c - FIND DECISION THRESHOLD ON VALIDATION DATASET - FDT
    MODEL_FIND_DECISION_THRES   = True
    WHAT_MODEL_TO_LOAD_FDT              = 'Best'  # If several models have been trained, which one to use. 'Best'=best model. or an integer to specify model number, e.g. 3 would load model 3.
    WHAT_MODEL_EPOCH_TO_LOAD_FDT        = 'Best'  # What epoch to load weights from. 'last'=last epoch, 'Best'=best epoch, or an integer to specify epoch, e.g. 120 would load weights from epoch 120.
    RUN_NEW_PREDICTION_FDT              = True  # True: Overwrite existing predictions. False=Load existing prediction from pickle if exist, else run new prediction.
    ######## DEBUG MODE #############################
    DEBUG_MODE_FDT                      = False  # Turn on debugging mode
    N_REGIONS_TO_PROCESS_DEBUG_MODE_FDT = 3
    ######## PREPROCESSING #####################################
    PRE_DIVIDE_TILES_BY_255_FDT         = True  # Preprocessing. Divide all pixel values by 255 to normalise values from 0-255 into 0-1.
    PRE_RGB_TO_BGR_FDT                  = True  # Preprocessing. Switch the color channels from RGB to BGR. (VGG16 pretrained weights are trained using BGR.)
    PRE_SUBTRACT_MEAN_FROM_TILE_FDT     = False  # Preprocessing. Subtract mean value from tiles. Mean values are stored in def dataset_preprocessing().
    # endregion

    # region MODE 9a - DIAGNOSTIC TEST SET (FOR REGIONS) - DTS
    MODEL_DIAGNOSTIC_TEST_SET   = True
    WHAT_MODEL_TO_LOAD_DTS              = 'Best'  # If several models have been trained, which one to use. 'Best'=best model. or an integer to specify model number, e.g. 3 would load model 3.
    WHAT_MODEL_EPOCH_TO_LOAD_DTS        = 'Best'  # What epoch to load weights from. 'last'=last epoch, 'Best'=best epoch, or an integer to specify epoch, e.g. 120 would load weights from epoch 120.
    TILES_TO_SHOW_DTS                   = ['400x']
    RUN_NEW_PREDICTION_DTS              = True  # True: Overwrite existing predictions. False=Load existing prediction from pickle if exist, else run new prediction.
    ######## DEBUG MODE #############################
    DEBUG_MODE_DTS                      = False  # Turn on debugging mode
    N_REGIONS_TO_PROCESS_DEBUG_MODE_DTS = 3
    ######## PREPROCESSING #####################################
    PRE_DIVIDE_TILES_BY_255_DTS         = True  # Preprocessing. Divide all pixel values by 255 to normalise values from 0-255 into 0-1.
    PRE_RGB_TO_BGR_DTS                  = True  # Preprocessing. Switch the color channels from RGB to BGR. (VGG16 pretrained weights are trained using BGR.)
    PRE_SUBTRACT_MEAN_FROM_TILE_DTS     = False  # Preprocessing. Subtract mean value from tiles. Mean values are stored in def dataset_preprocessing().
    # endregion

    # region SETTINGS AND PARAMETERS
    OPTIMIZER = ['SGD']  # Available: 'adam', 'adamax', 'nadam', 'adadelta', 'adagrad', 'SGD'.
    batch_size = 128  # Batch size, how many samples to include in each batch during training.
    ACTIVATE_ReduceLROnPlateau = True  # Reduce learning rate on plateau.
    USE_MULTIPROCESSING = False  # Use keras multiprocessing function. Gives warning, recommended to be False.
    CONTINUE_FROM_MODEL = 'last'  # If START_NEW_MODEL=False, this variable determines which model to continue working on. 'last'=continue last model, else specify model name eg. '2017-03-13_17-50-50'.
    N_WORKERS = 10
    MAX_QUEUE_SIZE = 32
    N_CHANNELS = 3
    TILE_SIZE_DIAGNOSTIC = 256
    START_NEW_MODEL = sys.argv[1]  # True=Start all new model from MODEL_MODE=0. False=Continue from previous
    tf_version = tf.__version__
    # endregion

    # region FOLDERS AND PATHS
    LOG_FOLDER = 'logs/'
    SAVED_DATA_FOLDER = 'Saved_data/'
    CLASSIFICATION_REPORT_FOLDER = 'Classification_reports/'
    MODE_DTL_FOLDER = 'Mode 08a - Diagnostic model/'
    MODE_FDT_FOLDER = 'Mode 08c - Decision threshold/'
    MODE_DIAGNOSTIC_TEST_SET_FOLDER = 'Mode 09a - Diagnostic test set/'
    ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE = 'ALL_MODEL_PARAMETERS_DTL_PICKLE.obj'
    MODELS_AND_LOSS_ARRAY_DTL_PICKLE_FILE = 'MODELS_AND_LOSS_ARRAY_DTL_PICKLE.obj'
    TRAINING_DATA_DTL_PICKLE_FILE = '_TRAINING_DATA_DTL_PICKLE.obj'

    # Model metadata folder (Put existing metadata folder here)
    METADATA_FOLDER = 'metadata/'

    # Model weight folder (Put weights for existing models here)
    MODEL_WEIGHT_FOLDER = 'Model_weights/'

    # Specify main dataset folder (Put all SCN files here in individual folders)
    SCN_PATH = 'WSI_files/'

    # Specify annotation mask folder (Put all coordinate pickle files here)
    DIAGNOSTIC_TRAINING_DICTS_PATH = 'Coordinate_dicts_files/diagnostic_training/'
    DIAGNOSTIC_VALIDATION_DICTS_PATH = 'Coordinate_dicts_files/diagnostic_validation/'
    DIAGNOSTIC_TEST_DICTS_PATH = 'Coordinate_dicts_files/diagnostic_test/'

    # endregion

    # region FILE_INITIALIZATION
    current_run_path, SUMMARY_CSV_FILE_PATH = my_functions.init_file(
        SAVED_DATA_FOLDER=SAVED_DATA_FOLDER,
        LOG_FOLDER=LOG_FOLDER,
        START_NEW_MODEL=START_NEW_MODEL,
        CONTINUE_FROM_MODEL=CONTINUE_FROM_MODEL,
        METADATA_FOLDER=METADATA_FOLDER,
        USE_MULTIPROCESSING=USE_MULTIPROCESSING)

    # endregion

    # region MODE 8a - DIAGNOSTIC LEARNING - (Finished)
    if TRAIN_DIAGNOSTIC_MODEL_DTL in [True, 'True', 'true']:
        my_functions.my_print('')
        my_functions.my_print('Mode 8a - Diagnostic transfer learning')

        # Load transfer learning function
        mode_8a_diagnostic_learning.transfer_learning_model(OPTIMIZER=OPTIMIZER,
                                                            batch_size=batch_size,
                                                            current_run_path=current_run_path,
                                                            METADATA_FOLDER=METADATA_FOLDER,
                                                            MODEL_WEIGHT_FOLDER=MODEL_WEIGHT_FOLDER,
                                                            N_CHANNELS=N_CHANNELS,
                                                            N_WORKERS=N_WORKERS,
                                                            EPOCHES=EPOCHES_DTL,
                                                            CLASSIFICATION_REPORT_FOLDER=CLASSIFICATION_REPORT_FOLDER,
                                                            layer_config=layer_config_DTL,
                                                            freeze_base_model=freeze_base_model_DTL,
                                                            learning_rate=learning_rate_DTL,
                                                            n_neurons_first_layer=n_neurons_first_layer_DTL,
                                                            n_neurons_second_layer=n_neurons_second_layer_DTL,
                                                            n_neurons_third_layer=n_neurons_third_layer_DTL,
                                                            dropout=dropout_DTL,
                                                            base_model=base_model_DTL,
                                                            MODE_FOLDER=MODE_DTL_FOLDER,
                                                            START_NEW_MODEL=START_NEW_MODEL,
                                                            EARLY_STOPPING_PATIENCE=EARLY_STOP_PATIENCE_DTL,
                                                            WHAT_LABELS_TO_USE=WHAT_LABELS_TO_USE_DTL,
                                                            base_model_pooling=base_model_pooling_DTL,
                                                            ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE=ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE,
                                                            MODELS_AND_LOSS_ARRAY_DTL_PICKLE_FILE=MODELS_AND_LOSS_ARRAY_DTL_PICKLE_FILE,
                                                            TRAINING_DATA_DTL_PICKLE_FILE=TRAINING_DATA_DTL_PICKLE_FILE,
                                                            REDUCE_LR_PATIENCE=REDUCE_LR_PATIENCE_DTL,
                                                            ACTIVATE_ReduceLROnPlateau=ACTIVATE_ReduceLROnPlateau,
                                                            USE_MULTIPROCESSING=USE_MULTIPROCESSING,
                                                            MAX_QUEUE_SIZE=MAX_QUEUE_SIZE,
                                                            SMALL_DATASET_DEBUG_MODE=SMALL_DATASET_DEBUG_MODE_DTL,
                                                            DIAGNOSTIC_TRAINING_DICTS_PATH=DIAGNOSTIC_TRAINING_DICTS_PATH,
                                                            which_model_mode_to_use=which_model_mode_to_use_DTL,
                                                            which_scale_to_use_mono=which_scale_to_use_mono_DTL,
                                                            which_scale_to_use_di=which_scale_to_use_di_DTL,
                                                            SCN_PATH=SCN_PATH,
                                                            SUMMARY_CSV_FILE_PATH=SUMMARY_CSV_FILE_PATH,
                                                            tf_version=tf_version,
                                                            TILE_SIZE_DIAGNOSTIC=TILE_SIZE_DIAGNOSTIC,
                                                            blocks_to_unfreeze_vgg16_vgg19=blocks_to_unfreeze_vgg16_vgg19_DTL,
                                                            delayed_unfreeze_start_epoche_vgg16_vgg19=delayed_unfreeze_start_epoche_vgg16_vgg19_DTL,
                                                            DIAGNOSTIC_VALIDATION_DICTS_PATH=DIAGNOSTIC_VALIDATION_DICTS_PATH,
                                                            SAVE_CONF_MAT_AND_REPORT_ON_END=SAVE_CONF_MAT_AND_REPORT_ON_END_DTL,
                                                            PRE_DIVIDE_TILES_BY_255=PRE_DIVIDE_TILES_BY_255_DTL,
                                                            PRE_RGB_TO_BGR=PRE_RGB_TO_BGR_DTL,
                                                            PRE_SUBTRACT_MEAN_FROM_TILE=PRE_SUBTRACT_MEAN_FROM_TILE_DTL,
                                                            BALANCE_DATASET_BY_AUGMENTATION=BALANCE_DATASET_BY_AUGMENTATION_DTL,
                                                            AUGMENT_MINORITY_WSI=AUGMENT_MINORITY_WSI_DTL,
                                                            DESIRED_TILES_IN_EACH_WSI=DESIRED_TILES_IN_EACH_WSI_DTL)
    # endregion

    # region MODE 8c - FIND DECISION THRESHOLD (Working on..)
    if MODEL_FIND_DECISION_THRES in [True, 'True', 'true']:
        my_functions.my_print('')
        my_functions.my_print('Mode 8c - Finding decision threshold on validation dataset')

        wsi_list_train = os.listdir(DIAGNOSTIC_TRAINING_DICTS_PATH)
        wsi_list_train.sort()
        total_no_of_wsi = len(wsi_list_train)

        # Loop through all WSI. Process one WSI at a time
        for current_wsi_index, current_wsi_coordinate_pickle_filename in enumerate(wsi_list_train):
            my_functions.my_print('')
            my_functions.my_print('Starting WSI {} of {} - Now processing {}'.format(str(current_wsi_index + 1), total_no_of_wsi,
                                                                   current_wsi_coordinate_pickle_filename.split("list")[0]))

            wsi_filename_no_extension = current_wsi_coordinate_pickle_filename.split("list")[0][:-1]
            wsi_filename_w_extension = wsi_filename_no_extension + '.scn'
            wsi_dataset_folder = SCN_PATH + wsi_filename_no_extension + '/'
            wsi_dataset_file_path = wsi_dataset_folder + wsi_filename_w_extension

            # Run prediction model
            current_model_name = mode_8c_find_decision_threshold.predict_validation_dataset(current_run_path=current_run_path,
                                                                           METADATA_FOLDER=METADATA_FOLDER,
                                                                           MODEL_WEIGHT_FOLDER=MODEL_WEIGHT_FOLDER,
                                                                           N_CHANNELS=N_CHANNELS,
                                                                           WHAT_MODEL_TO_LOAD=WHAT_MODEL_TO_LOAD_FDT,
                                                                           WHAT_MODEL_EPOCH_TO_LOAD=WHAT_MODEL_EPOCH_TO_LOAD_FDT,
                                                                           MODE_FOLDER=MODE_FDT_FOLDER,
                                                                           DIAGNOSTIC_TRAINING_DICTS_PATH=DIAGNOSTIC_TRAINING_DICTS_PATH,
                                                                           SCN_PATH=SCN_PATH,
                                                                           MODELS_AND_LOSS_ARRAY_DTL_PICKLE_FILE=MODELS_AND_LOSS_ARRAY_DTL_PICKLE_FILE,
                                                                           ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE=ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE,
                                                                           TRAINING_DATA_DTL_PICKLE_FILE=TRAINING_DATA_DTL_PICKLE_FILE,
                                                                           TILE_SIZE_DIAGNOSTIC=TILE_SIZE_DIAGNOSTIC,
                                                                           current_wsi_coordinate_pickle_filename=current_wsi_coordinate_pickle_filename,
                                                                           wsi_filename_no_extension=wsi_filename_no_extension,
                                                                           DEBUG_MODE=DEBUG_MODE_FDT,
                                                                           MAX_QUEUE_SIZE=MAX_QUEUE_SIZE,
                                                                           N_WORKERS=N_WORKERS,
                                                                           USE_MULTIPROCESSING=USE_MULTIPROCESSING,
                                                                           N_REGIONS_TO_PROCESS_DEBUG_MODE=N_REGIONS_TO_PROCESS_DEBUG_MODE_FDT,
                                                                           PRE_DIVIDE_TILES_BY_255=PRE_DIVIDE_TILES_BY_255_FDT,
                                                                           PRE_RGB_TO_BGR=PRE_RGB_TO_BGR_FDT,
                                                                           PRE_SUBTRACT_MEAN_FROM_TILE=PRE_SUBTRACT_MEAN_FROM_TILE_FDT,
                                                                           RUN_NEW_PREDICTION=RUN_NEW_PREDICTION_FDT)

        mode_8c_find_decision_threshold.find_wsi_decision_thresholds(SCN_PATH=SCN_PATH,
                                                                     DIAGNOSTIC_TRAINING_DICTS_PATH=DIAGNOSTIC_TRAINING_DICTS_PATH,
                                                                     current_run_path=current_run_path,
                                                                     MODE_FOLDER=MODE_FDT_FOLDER,
                                                                     current_model_name=current_model_name,
                                                                     DEBUG_MODE=DEBUG_MODE_FDT,
                                                                     N_REGIONS_TO_PROCESS_DEBUG_MODE=N_REGIONS_TO_PROCESS_DEBUG_MODE_FDT)

    # endregion

    # region MODE 9a - DIAGNOSTIC TEST SET (FOR REGIONS) (Finished)
    if MODEL_DIAGNOSTIC_TEST_SET in [True, 'True', 'true']:
        my_functions.my_print('')
        my_functions.my_print('Mode 9a - Diagnostic test set (for regions)')

        wsi_list_test = os.listdir(DIAGNOSTIC_TEST_DICTS_PATH)
        wsi_list_test.sort()
        total_no_of_wsi = len(wsi_list_test)

        # Loop through all WSI. Process one WSI at a time
        for current_wsi_index, current_wsi_coordinate_pickle_filename in enumerate(wsi_list_test):
            my_functions.my_print('')
            my_functions.my_print('Starting WSI {} of {} - Now processing {}'.format(str(current_wsi_index + 1), total_no_of_wsi,
                                                                                     current_wsi_coordinate_pickle_filename.split("list")[0]))

            wsi_filename_no_extension = current_wsi_coordinate_pickle_filename.split("list")[0][:-1]
            wsi_filename_w_extension = wsi_filename_no_extension + '.scn'
            wsi_dataset_folder = SCN_PATH + wsi_filename_no_extension + '/'
            wsi_dataset_file_path = wsi_dataset_folder + wsi_filename_w_extension

            # Run prediction model
            current_model_name, N_CLASSES_TRAINING, NAME_OF_CLASSES_TRAINING_DISPLAYNAME, \
            NAME_OF_CLASSES_TRAINING = mode_9a_diagnostic_test_set.diagnostic_test_set(
                current_run_path=current_run_path,
                METADATA_FOLDER=METADATA_FOLDER,
                MODEL_WEIGHT_FOLDER=MODEL_WEIGHT_FOLDER,
                N_CHANNELS=N_CHANNELS,
                WHAT_MODEL_TO_LOAD=WHAT_MODEL_TO_LOAD_DTS,
                WHAT_MODEL_EPOCH_TO_LOAD=WHAT_MODEL_EPOCH_TO_LOAD_DTS,
                MODE_FOLDER=MODE_DIAGNOSTIC_TEST_SET_FOLDER,
                DIAGNOSTIC_TEST_DICTS_PATH=DIAGNOSTIC_TEST_DICTS_PATH,
                SCN_PATH=SCN_PATH,
                MODELS_AND_LOSS_ARRAY_DTL_PICKLE_FILE=MODELS_AND_LOSS_ARRAY_DTL_PICKLE_FILE,
                ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE=ALL_MODEL_PARAMETERS_DTL_PICKLE_FILE,
                TRAINING_DATA_DTL_PICKLE_FILE=TRAINING_DATA_DTL_PICKLE_FILE,
                SUMMARY_CSV_FILE_PATH=SUMMARY_CSV_FILE_PATH,
                tf_version=tf_version,
                TILE_SIZE_DIAGNOSTIC=TILE_SIZE_DIAGNOSTIC,
                current_wsi_coordinate_pickle_filename=current_wsi_coordinate_pickle_filename,
                wsi_filename_no_extension=wsi_filename_no_extension,
                wsi_dataset_folder=wsi_dataset_folder,
                wsi_dataset_file_path=wsi_dataset_file_path,
                TILES_TO_SHOW=TILES_TO_SHOW_DTS,
                DEBUG_MODE=DEBUG_MODE_DTS,
                MAX_QUEUE_SIZE=MAX_QUEUE_SIZE,
                N_WORKERS=N_WORKERS,
                USE_MULTIPROCESSING=USE_MULTIPROCESSING,
                N_REGIONS_TO_PROCESS_DEBUG_MODE=N_REGIONS_TO_PROCESS_DEBUG_MODE_DTS,
                PRE_DIVIDE_TILES_BY_255=PRE_DIVIDE_TILES_BY_255_DTS,
                PRE_RGB_TO_BGR=PRE_RGB_TO_BGR_DTS,
                PRE_SUBTRACT_MEAN_FROM_TILE=PRE_SUBTRACT_MEAN_FROM_TILE_DTS,
                RUN_NEW_PREDICTION=RUN_NEW_PREDICTION_DTS,
                current_wsi_index=current_wsi_index,
                MODE_FDT_FOLDER=MODE_FDT_FOLDER)

            time.sleep(10)

        mode_9a_diagnostic_test_set.diagnostic_test_final_score(current_run_path=current_run_path,
                                                                MODE_FOLDER=MODE_DIAGNOSTIC_TEST_SET_FOLDER,
                                                                current_model_name=current_model_name,
                                                                N_CLASSES_TRAINING=N_CLASSES_TRAINING,
                                                                NAME_OF_CLASSES_TRAINING_DISPLAYNAME=NAME_OF_CLASSES_TRAINING_DISPLAYNAME,
                                                                NAME_OF_CLASSES_TRAINING=NAME_OF_CLASSES_TRAINING)

    # endregion
