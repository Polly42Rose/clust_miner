import pm4py
from collections import Counter
import os


def get_attributes(log):
    attr_counter = Counter()
    for trace in log:
        for event in trace:
            for attr in event:
                attr_counter[attr] += 1
    attributes = list()
    for attr in attr_counter:
        if attr_counter[attr] > 1:
            attributes.append(attr)
    return attributes


def parse_log(file):
    ext = os.path.splitext(file.name)[1]
    if ext != ".xes":
        raise NameError("Неверный формат файла! Загрузите файл формата .xes.")
    with open('temp.xes', 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    try:
        log = pm4py.read_xes("temp.xes")
    except:
        raise OSError("Ошибка в файле!")
    os.remove("temp.xes")
    attr_list = get_attributes(log)
    attributes = ""
    for i in range(len(attr_list)):
        attributes += attr_list[i]
        if i != len(attr_list) - 1:
            attributes += ", "
    n_traces = len(log)
    return log, n_traces, attributes
