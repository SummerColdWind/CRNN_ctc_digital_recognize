import yaml

def get_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    global_config, train_config, test_config = config['Global'], config['Train'], config['Test']
    return global_config, train_config, test_config
