import numpy as np

# map vips formats to np dtypes
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

# Conversion factors between different magnification scale of SCN images
# To go UP in magnification level from (25x to 100x) or (25x to 400x) or (100x to 400x) use "divide"
# To go DOWN in magnification level from (100x to 25x) or (400x to 25x) or (400x to 100x) use "multiply"
Scale_between_100x_400x = 0.25
Scale_between_25x_100x = 0.25
Scale_between_25x_400x = 0.0625

def get_diagnostic_name_and_index_of_classes(label_to_use):
    """
    Info:
        WHO73: Grade1=0, Grade2=1, Grade3=2.
        WHO04: HighGrade=0, LowGrade=1.
        Stage: TA=0, T1=1.
        Recurrence: NoRecurrence=0, YesRecurrence=1.
        Progression: NoProgression=0, YesProgression=1.
        WHOcombined: LowGrade+Grade1=0, LowGrade+Grade2=1, HighGrade+Grade2=2, HighGrade+Grade3=3
    """
    name_and_index_of_classes = dict()
    if label_to_use == 'WHO73':
        name_and_index_of_classes[0] = {'display_name': 'Grade 1', 'name': 'grade1', 'index': 0, 'used_in_training': 1}
        name_and_index_of_classes[1] = {'display_name': 'Grade 2', 'name': 'grade2', 'index': 1, 'used_in_training': 1}
        name_and_index_of_classes[2] = {'display_name': 'Grade 3', 'name': 'grade3', 'index': 2, 'used_in_training': 1}
        name_and_index_of_classes[3] = {'display_name': 'Undefined', 'name': 'undefined', 'index': 3, 'used_in_training': 0}
    elif label_to_use == 'WHO04':
        name_and_index_of_classes[0] = {'display_name': 'High Grade', 'name': 'high_grade', 'index': 0, 'used_in_training': 1}
        name_and_index_of_classes[1] = {'display_name': 'Low Grade', 'name': 'low_grade', 'index': 1, 'used_in_training': 1}
        name_and_index_of_classes[2] = {'display_name': 'Undefined', 'name': 'undefined', 'index': 2, 'used_in_training': 0}
    elif label_to_use == 'Stage':
        name_and_index_of_classes[0] = {'display_name': 'TA', 'name': 'ta', 'index': 0, 'used_in_training': 1}
        name_and_index_of_classes[1] = {'display_name': 'T1', 'name': 't1', 'index': 1, 'used_in_training': 1}
        name_and_index_of_classes[2] = {'display_name': 'Undefined', 'name': 'undefined', 'index': 2, 'used_in_training': 0}
    if label_to_use == 'Recurrence':
        name_and_index_of_classes[0] = {'display_name': 'No Recurrence', 'name': 'no_recurrence', 'index': 0, 'used_in_training': 1}
        name_and_index_of_classes[1] = {'display_name': 'Recurrence', 'name': 'recurrence', 'index': 1, 'used_in_training': 1}
        name_and_index_of_classes[2] = {'display_name': 'Undefined', 'name': 'undefined', 'index': 2, 'used_in_training': 0}
    elif label_to_use == 'Progression':
        name_and_index_of_classes[0] = {'display_name': 'No Progression', 'name': 'no_progression', 'index': 0, 'used_in_training': 1}
        name_and_index_of_classes[1] = {'display_name': 'Progression', 'name': 'progression', 'index': 1, 'used_in_training': 1}
        name_and_index_of_classes[2] = {'display_name': 'Undefined', 'name': 'undefined', 'index': 2, 'used_in_training': 0}
    elif label_to_use == 'WHOcombined':
        name_and_index_of_classes[0] = {'display_name': '0', 'name': '0', 'index': 0, 'used_in_training': 1}
        name_and_index_of_classes[1] = {'display_name': '1', 'name': '1', 'index': 1, 'used_in_training': 1}
        name_and_index_of_classes[2] = {'display_name': '2', 'name': '2', 'index': 2, 'used_in_training': 1}
        name_and_index_of_classes[3] = {'display_name': '3', 'name': '3', 'index': 3, 'used_in_training': 1}
        name_and_index_of_classes[4] = {'display_name': 'Undefined', 'name': 'undefined', 'index': 4, 'used_in_training': 0}

    return name_and_index_of_classes
