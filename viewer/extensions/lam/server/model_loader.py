import os
import re
import io
import sys
import glob
import math
from os import path as osp
from basicsr.train import build_model
from contextlib import contextmanager
from basicsr.utils.options import parse_options
import traceback
import basicsr
import yaml

root_path = osp.dirname(osp.dirname(basicsr.__file__))
sys.path.append(root_path)
# os.chdir(root_path)

def get_root_path():
    return root_path

def str2dict(input_str):
    def parse_value(value):
        if value.startswith('[') and value.endswith(']'):
            items = value[1:-1].split(',')
            return [parse_individual_item(item.strip()) for item in items]

        return parse_individual_item(value)

    def parse_individual_item(item):
        item = item.strip()
        if item.startswith("'") and item.endswith("'"):
            item = item[1:-1]
        elif item.startswith('"') and item.endswith('"'):
            item = item[1:-1]

        if item.lower() == 'true':
            return True
        elif item.lower() == 'false':
            return False
        else:
            try:
                return int(item)
            except ValueError:
                try:
                    return float(item)
                except ValueError:
                    return item

    def add_to_dict(stack, key, value):
        if isinstance(stack[-1], list):
            stack[-1].append((key, value))
        elif isinstance(stack[-1], dict):
            stack[-1][key] = value

    lines = input_str.split('\n')
    stack = [{}]

    for line in lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        level = indent // 2 - 1

        if line.strip().endswith('['):
            key = line.strip()[:-2].strip()
            new_dict = {}
            add_to_dict(stack, key, new_dict)
            stack.append(new_dict)
        elif line.strip() == ']':
            completed_dict = stack.pop()
            if isinstance(stack[-1], list):
                stack[-1].append(completed_dict)
        else:
            key, value = line.strip().split(': ', 1)
            value = parse_value(value)
            add_to_dict(stack, key, value)

    return stack[0]

def extract_ymal_string_from_log(filepath):
    with open(filepath, 'r') as file:
        return file.read()

def save_dict_as_yaml(parsed_dict, filepath):
    with open(filepath, 'w') as file:
        yaml.dump(parsed_dict, file, default_flow_style=False, sort_keys=False)

@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = original_stdout

def get_logs():
    results_path = os.path.join(root_path, 'results')
    log_files = []
    for dirpath, dirnames, filenames in os.walk(results_path):
        sorted_logs = sorted([file for file in filenames if file.endswith('.log')])
        if sorted_logs:
            log_files.append(os.path.join(dirpath, sorted_logs[-1]))
        if sorted_logs:
            dirnames.clear()
    return log_files

logs = get_logs()
def get_model_log(path):
    global logs
    for log in logs:
        if osp.dirname(log).endswith(path):
            return log
    logs = get_logs()
    for log in logs:
        if osp.dirname(log).endswith(path):
            return log


def extract_yaml_from_log(log_path):
    with open(log_path, 'r') as file:
        log_content = file.read()

    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO: \n([\s\S]*?)(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    matches = re.findall(pattern, log_content)

    for match in matches:
        if "name: " in match:
            return match

argv = sys.argv.copy()
def get_model(opt_path):
    original_path = os.getcwd()
    os.chdir(root_path)

    sys.argv = [argv[0], '-opt', opt_path]

    opt, args = None, None
    with suppress_stdout():
        opt, args = parse_options(root_path, False)
    opt['root_path'] = root_path

    model = None
    try:
        model = build_model(opt)
    except Exception as error:
        print(opt_path, 'build_model error.', error)
        traceback.print_exc()
    # net = model.get_bare_model(model.net_g)
    # parameters = sum(map(lambda x: x.numel(), net.parameters()))
    os.chdir(original_path)
    return model

def get_model_from_path(path):
    log_path = get_model_log(path)
    yaml_path = log_path + '.yml'
    save_dict_as_yaml(str2dict(extract_yaml_from_log(log_path)), yaml_path)
    return get_model(yaml_path)
