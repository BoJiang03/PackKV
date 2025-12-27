
def merge(dict1, dict2):
    rt_map = {}
    for (hash, data) in dict1.items():
        rt_map[hash] = data

    for (hash, data) in dict2.items():
        if hash not in rt_map:
            rt_map[hash] = data
        else:
            assert rt_map[hash] == data, "hash: {}".format(hash)

    return rt_map

def key_concat(dict1, dict2):
    rt_list = []
    for (hash, data) in dict1.items():
        if hash in dict2:
            rt_list.append((data, dict2[hash]))

    return rt_list

