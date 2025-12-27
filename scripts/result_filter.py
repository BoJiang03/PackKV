from utils.serialization import load, save

setting_path = "data/accuracy/accuracy_setting_map.pkl"
setting_map = load(setting_path)

result_path = "data/accuracy/accuracy_result_map.pkl"
result_map = load(result_path)

new_result_map = {}
for benchmark, config in setting_map.items():
    if benchmark in result_map:
        new_result_map[benchmark] = result_map[benchmark]

new_result_path = "data/accuracy/accuracy_result_map.pkl"
save(new_result_map, new_result_path)
# print(new_result_map)
# for benchmark, result in new_result_map.items():
#     setting = setting_map[benchmark]
#     print(setting)
#     print(result)
print(f"Found {len(new_result_map)} results reusable")