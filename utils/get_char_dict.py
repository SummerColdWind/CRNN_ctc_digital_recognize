def get_char_dict(dict_path):
    with open(dict_path, 'r') as file:
        return [' '] + [char.strip() for char in file.readlines()]
