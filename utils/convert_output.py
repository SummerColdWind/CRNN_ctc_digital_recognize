import itertools

def convert_test_output(output, char_dict):
    result = output.max(axis=1).indices
    predict = ''.join([char_dict[num] for num, _ in itertools.groupby(result.tolist()) if num != 0])
    return predict

def convert_infer_output(output, char_dict):
    output = output[:, 0, :]
    return convert_test_output(output, char_dict)





