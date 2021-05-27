import tempfile

import pandas as pd
import numpy as np
from graphviz import Digraph
from sklearn.cluster import KMeans
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from collections import Counter, defaultdict
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.obj import Event
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from sklearn.metrics import silhouette_score
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
import datetime

LOG_FILE = "InternationalDeclarations.xes"
ACTIVITY_ID = "concept:name"
TRACE_ID = "trace_id"
TIMESTAMP_ID = "time:timestamp"
# not orange, green, black
COLORS = ["blue", "aquamarine", "antiquewhite", "coral1", "mistyrose",
          "brown1", "cadetblue1", "chartreuse1", "bisque4", "darkgoldenrod1",
          "darkgrey", "darkkhaki", "darkolivegreen3", "darkseagreen1", "deeppink",
          "deepskyblue", "gold", "lightpink", "lightsalmon", "lightseagreen",
          "lightskyblue3", "lightyellow1", "lightsteelblue1", "blueviolet", "moccasin"]


# read log to DataFrame
def read_log(xes_file):
    log = pm4py.read.read_xes(xes_file)
    return log


# creates df of log traces
def create_log_df(log):
    attr_counter = Counter()
    for trace in log:
        for event in trace:
            for attr in event:
                attr_counter[attr] += 1
    data = dict()
    attributes = list()
    for attr in attr_counter:
        if attr_counter[attr] > 1:
            attributes.append(attr)
            data[attr] = []
    data["trace_id"] = []
    for trace_id, trace in enumerate(log):
        for event in trace:
            for attr in data:
                if attr == "trace_id":
                    continue
                if attr in event:
                    data[attr].append(event[attr])
                else:
                    data[attr].append(np.nan)
            data["trace_id"].append(trace_id)

    df = pd.DataFrame(data=data)
    return df


# inductive miner
def inductive_miner(log):
    petri_net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log)
    return petri_net, initial_marking, final_marking


def mine_model_for_clusters(cluster_logs):
    models = []
    for log in cluster_logs:
        petri_net, initial_marking, final_marking = inductive_miner(log)
        models.append((petri_net, initial_marking, final_marking))
    return models


def visualize_petri_net(net, initial_marking, final_marking):
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    return gviz


def visualize_petri_net_color_clusters(net, initial_marking, final_marking, activity_to_cluster):
    decorations = dict()
    for transition in net.transitions:
        if transition.label in activity_to_cluster:
            cluster_id = activity_to_cluster[transition.label]
            decorations[transition] = {"label": transition.label, "color": COLORS[cluster_id]}

    gviz = graphviz_visualization(net, initial_marking=initial_marking, final_marking=final_marking,
                                  decorations=decorations)
    return gviz


def save_visualization(gviz, file):
    pn_visualizer.save(gviz, file)


# convert df to log format
def convert_df_to_log(df):
    df = df.sort_values('time:timestamp')
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'trace_id'}
    event_log = log_converter.apply(df, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
    return event_log


def get_activity_df(df):
    """
    Creates activity count dataframe from log dataframe. Throws out id-kind activities.
    :param df: log dataframe
    :return: activity count dataframe
    """
    df_analysis = df.copy()
    actions = df[ACTIVITY_ID].unique()
    n_events = len(df)

    # those are kinds of id or non important
    for col in df_analysis.columns:
        if len(df_analysis[col].unique()) >= min(n_events, 1000) or \
                len(df[col].value_counts()) == 1 or col == TIMESTAMP_ID:
            if col == ACTIVITY_ID:
                continue
            df_analysis.drop([col], axis=1, inplace=True)
    if "trace_id" in df_analysis.columns:
        df_analysis.drop(["trace_id"], axis=1, inplace=True)
    df_actions = pd.DataFrame(data={"Activity": actions})  # count df

    for col in df_analysis.columns:
        if col == ACTIVITY_ID:
            continue
        attr_values = df_analysis[col].unique()
        for val in attr_values:
            df_actions[col + ": " + str(val)] = 0

    df_actions = df_actions.set_index("Activity")

    for col in df_analysis.columns:
        if col == ACTIVITY_ID:
            continue
        if col == TRACE_ID:
            continue
        for action in df_actions.index:
            val_dict = dict(df_analysis[df_analysis[ACTIVITY_ID] == action][col].value_counts())
            for k, v in val_dict.items():
                df_actions.loc[action, col + ": " + k] = v

    return df_actions


def get_cluster_activities(cluster_labels, df_actions):
    """

    :param cluster_labels: cluster labels of activities by index
    :param df_actions: actions attribute count dataframe
    :return: list of activities for each cluster, cluster to activities map, activity to cluster map
    """
    cluster_to_activities = defaultdict(list)
    activity_to_cluster = dict()
    clusters = sorted(list(set(cluster_labels)))
    cluster_activities_list = []
    for clust_label in clusters:
        cluster_activities = df_actions.iloc[np.where(cluster_labels == clust_label)[0]]["Activity"]
        cluster_activities = list(cluster_activities)
        for activity in cluster_activities:
            cluster_to_activities[clust_label].append(activity)
            activity_to_cluster[activity] = clust_label
        cluster_activities_list.append(cluster_activities)
    return cluster_activities_list, cluster_to_activities, activity_to_cluster


def add_cluster_bounds(log, activity_to_cluster, cluster_activities, n_clust):
    for i in range(n_clust):
        transition_name = f"START_CLUSTER_{i}"
        cluster_activities[i].append(transition_name)
        activity_to_cluster[transition_name] = i
        transition_name = f"END_CLUSTER_{i}"
        cluster_activities[i].append(transition_name)
        activity_to_cluster[transition_name] = i

    for i in range(len(log)):
        # add starts
        trace = []
        visited = [False] * n_clust
        for j in range(len(log[i])):
            cluster = activity_to_cluster[log[i][j][ACTIVITY_ID]]
            if not visited[cluster]:
                visited[cluster] = True
                event_dict = Event({ACTIVITY_ID: f"START_CLUSTER_{cluster}",
                                    TIMESTAMP_ID: log[i][j][TIMESTAMP_ID] + datetime.timedelta(microseconds=-1)})
                trace.append(event_dict)
            trace.append(log[i][j])
        # add ends
        final_trace = []
        visited = [False] * n_clust
        for event in trace[::-1]:
            cluster = activity_to_cluster[event[ACTIVITY_ID]]
            if not visited[cluster]:
                visited[cluster] = True
                event_dict = Event({ACTIVITY_ID: f"END_CLUSTER_{cluster}",
                                    TIMESTAMP_ID: event[TIMESTAMP_ID] + datetime.timedelta(microseconds=1)})
                final_trace.append(event_dict)
            final_trace.append(event)
            final_trace.reverse()
        log[i]._list = final_trace
    return log


def expand_clusters(log, activity_to_cluster, n_clusters):
    dfg = dfg_discovery.apply(log)
    cluster_connections = list()
    embedding_graph = dict()
    for n in range(n_clusters):
        cluster_connections.append([0] * n_clusters)
    for cluster in range(n_clusters):
        embedding_graph[cluster] = []
    for pos, count in dfg.items():
        if pos[1][:-1] != "START_CLUSTER_":
            continue
        cluster_0 = activity_to_cluster[pos[0]]
        cluster_1 = activity_to_cluster[pos[1]]
        if cluster_0 != cluster_1:
            cluster_connections[cluster_1][cluster_0] += 1
    for n in range(n_clusters):
        temp_max = max(cluster_connections[n])
        candidate_parent = np.argmax(cluster_connections[n])
        if temp_max == 0:
            continue
        if candidate_parent not in embedding_graph[n]:
            embedding_graph[candidate_parent].append(n)
        else:
            reverse_max = cluster_connections[candidate_parent][n]
            if temp_max > reverse_max:
                embedding_graph[n].remove(candidate_parent)
                embedding_graph[candidate_parent].append(n)

    return embedding_graph


def mix_clusters(log, activity_to_cluster, cluster_activities):
    cycle = True
    embedding_graph = dict()
    while cycle:
        n_clusters = len(cluster_activities)
        concats = dict()
        embedding_graph = expand_clusters(log, activity_to_cluster, n_clusters)
        to_skip = []
        for i in range(len(embedding_graph)):
            if i in to_skip:
                continue
            for j in embedding_graph[i]:
                if j in to_skip:
                    continue
                if i in embedding_graph[j]:
                    to_skip.append(j)
                    concats[i] = j
                    concats[j] = i
        if len(concats) == 0:
            cycle = False
        else:
            new_cluster_activities = []
            passed = []
            for j in range(len(cluster_activities)):
                if j not in concats:
                    new_cluster_activities.append(cluster_activities[j])
                else:
                    passed.append(j)
            while passed:
                clust0 = passed[0]
                clust1 = concats[clust0]
                passed.remove(clust0)
                passed.remove(clust1)
                new_cluster_activities.append(cluster_activities[clust0] + cluster_activities[clust1])
            cluster_activities = new_cluster_activities
            activity_to_cluster = dict()
            for k in range(len(cluster_activities)):
                for activity in cluster_activities[k]:
                    activity_to_cluster[activity] = k

    for i in range(len(cluster_activities)):
        for j in embedding_graph[i]:
            cluster_activities[i].append(f"START_CLUSTER_{j}")
    return embedding_graph


def filter_log(cluster_activities_list, df):
    """
    filters log by cluster
    :param cluster_activities_list: list of activities for each cluster
    :param df: log dataframe
    :return: list of logs
    """
    clust_logs = []
    for cluster_activities in cluster_activities_list:
        clust_df = df[df[ACTIVITY_ID].isin(cluster_activities)]
        clust_log = convert_df_to_log(clust_df)
        clust_logs.append(clust_log)
    return clust_logs


def best_n_clust(X, algo):
    n = 1
    max_score = 0
    max_n_clust = min(25, X.shape[0])
    for n_clust in range(2, max_n_clust):
        labels = algo(X, n_clust)
        try:
            score = silhouette_score(X, labels)
        except:
            continue
        if score > max_score:
            max_score = score
            n = n_clust
    return n, max_score


def k_means(X, n_clust):
    kmeans = KMeans(n_clust)
    kmeans.fit(X)
    return kmeans.predict(X)


def add_parent_model(net, pm, cluster_number, activity_to_cluster,
                     name_to_place, name_to_transition, root):
    for place in pm.places:
        if place.name == "source" or place.name == "sink":
            if not root:
                continue
            else:
                name = place.name
        else:
            name = f"{place.name}_cl{cluster_number}"
        p = PetriNet.Place(name)
        net.places.add(p)
        name_to_place[name] = p

    for transition in pm.transitions:
        if transition.label in activity_to_cluster \
                and activity_to_cluster[transition.label] != cluster_number:
            continue
        if transition.label in activity_to_cluster:
            name = f"{transition.label}"
            t = PetriNet.Transition(name, name)
        else:
            name = f"{transition.name}_cl{cluster_number}"
            t = PetriNet.Transition(name, transition.label)
        net.transitions.add(t)
        name_to_transition[name] = t

    for arc in pm.arcs:
        if not root and ((type(arc.source) == PetriNet.Place and arc.source.name == "source")
                         or (type(arc.target) == PetriNet.Place and arc.target.name == "sink")):
            continue

        if type(arc.source) == PetriNet.Place:
            if arc.source.name == "source":
                p1 = name_to_place[arc.source.name]
            else:
                p1 = name_to_place[f"{arc.source.name}_cl{cluster_number}"]
        else:
            if arc.source.label in name_to_transition:
                if arc.source.label in activity_to_cluster \
                        and activity_to_cluster[arc.source.label] != cluster_number:
                    p1 = name_to_transition[f"END_CLUSTER_{activity_to_cluster[arc.source.label]}"]
                else:
                    p1 = name_to_transition[arc.source.label]
            else:
                p1 = name_to_transition[f"{arc.source.name}_cl{cluster_number}"]
        if type(arc.target) == PetriNet.Place:
            if arc.target.name == "sink":
                p2 = name_to_place[arc.target.name]
            else:
                p2 = name_to_place[f"{arc.target.name}_cl{cluster_number}"]
        else:
            if arc.target.label in name_to_transition:
                p2 = name_to_transition[arc.target.label]
            else:
                p2 = name_to_transition[f"{arc.target.name}_cl{cluster_number}"]
        petri_utils.add_arc_from_to(p1, p2, net)


# merges models from different clusters
def merge_models(cluster_pm, activity_to_cluster, embedding_graph):
    net = PetriNet("merged_petri_net")
    name_to_place = dict()
    name_to_transition = dict()
    root = [True] * len(embedding_graph)

    for k, v in embedding_graph.items():
        for node in v:
            root[node] = False

    visited = [False] * len(embedding_graph)

    def embed_models(parent):
        if visited[parent]:
            return
        children = embedding_graph[parent]
        if len(children) == 0:
            add_parent_model(net, cluster_pm[parent][0], parent,
                             activity_to_cluster, name_to_place, name_to_transition, root[parent])
            visited[parent] = True
        else:
            for child in children:
                if not visited[child]:
                    embed_models(child)
            add_parent_model(net, cluster_pm[parent][0], parent,
                             activity_to_cluster, name_to_place, name_to_transition, root[parent])
            visited[parent] = True

    for key in embedding_graph:
        embed_models(key)

    to_del_transitions = []
    to_del_arcs = []
    for transition in net.transitions:
        if len(transition.in_arcs) == 0:
            to_del_transitions.append(transition)
            for arc in transition.out_arcs:
                to_del_arcs.append(arc)
        if len(transition.out_arcs) == 0:
            to_del_transitions.append(transition)
            for arc in transition.in_arcs:
                to_del_arcs.append(arc)
    for el in to_del_transitions:
        net.transitions.remove(el)
    for arc in to_del_arcs:
        net.arcs.remove(arc)

    has_parent = [False] * len(embedding_graph)
    for key in embedding_graph:
        for child in embedding_graph[key]:
            has_parent[child] = True

    initial_marking = Marking()
    initial_marking[name_to_place["source"]] = 1
    final_marking = Marking()
    final_marking[name_to_place["sink"]] = 1

    for key in embedding_graph:
        if len(embedding_graph[key]) == 0 and not has_parent[key]:
            start_cluster = name_to_transition[f"START_CLUSTER_{key}"]
            source = name_to_place["source"]
            added = False
            for el in source.out_arcs:
                if el.target == start_cluster:
                    added = True
                    break
            if not added:
                for arc in start_cluster.in_arcs:
                    net.places.remove(arc.source)
                    net.arcs.remove(arc)
                source = PetriNet.Place(f"source_cl{key}")
                net.places.add(source)
                initial_marking[source] = 1
                petri_utils.add_arc_from_to(source, start_cluster, net)
            end_cluster = name_to_transition[f"END_CLUSTER_{key}"]
            sink = name_to_place["sink"]
            added = False
            for el in sink.in_arcs:
                if el.source == end_cluster:
                    added = True
                    break
            if not added:
                for arc in end_cluster.out_arcs:
                    net.places.remove(arc.target)
                    net.arcs.remove(arc)
                sink = PetriNet.Place(f"sink_cl{key}")
                net.places.add(sink)
                final_marking[sink] = 1
                petri_utils.add_arc_from_to(end_cluster, sink, net)

    return net, initial_marking, final_marking


def run_pipline(xes_file):
    log = read_log(xes_file)
    df = create_log_df(log)
    df_actions = get_activity_df(df)
    X = np.array(df_actions)
    n_clust, sil_score = best_n_clust(X, k_means)
    clusters = k_means(X, n_clust)

    df_actions = df_actions.reset_index()
    cluster_activities, cluster_to_activities, activity_to_cluster = get_cluster_activities(clusters, df_actions)
    log = add_cluster_bounds(log, activity_to_cluster, cluster_activities, n_clust)
    embedding_graph = mix_clusters(log, activity_to_cluster, cluster_activities)
    df = create_log_df(log)
    cluster_logs = filter_log(cluster_activities, df)
    cluster_pm = mine_model_for_clusters(cluster_logs)
    net, im, fm = merge_models(cluster_pm, activity_to_cluster, embedding_graph)
    return net, im, fm, activity_to_cluster


def graphviz_visualization(net, image_format="png", initial_marking=None, final_marking=None, decorations=None,
                           debug=False, set_rankdir=None, font_size="12"):
    """
    Provides visualization for the petrinet

    Parameters
    ----------
    net: :class:`pm4py.entities.petri.petrinet.PetriNet`
        Petri net
    image_format
        Format that should be associated to the image
    initial_marking
        Initial marking of the Petri net
    final_marking
        Final marking of the Petri net
    decorations
        Decorations of the Petri net (says how element must be presented)
    debug
        Enables debug mode
    set_rankdir
        Sets the rankdir to LR (horizontal layout)

    Returns
    -------
    viz :
        Returns a graph object
    """
    if initial_marking is None:
        initial_marking = Marking()
    if final_marking is None:
        final_marking = Marking()
    if decorations is None:
        decorations = {}

    font_size = str(font_size)

    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    viz = Digraph(net.name, filename=filename.name, engine='dot', graph_attr={'bgcolor': 'transparent'})
    if set_rankdir:
        viz.graph_attr['rankdir'] = set_rankdir
    else:
        viz.graph_attr['rankdir'] = 'LR'

    # transitions
    viz.attr('node', shape='box')
    for t in net.transitions:
        if t.label is not None:
            if t in decorations and "label" in decorations[t] and "color" in decorations[t]:
                viz.node(str(id(t)), decorations[t]["label"], style='filled', fillcolor=decorations[t]["color"],
                         border='1', fontsize=font_size)
            else:
                viz.node(str(id(t)), str(t.label), fontsize=font_size)
        else:
            if debug:
                viz.node(str(id(t)), str(t.name), fontsize=font_size)
            elif t in decorations and "color" in decorations[t] and "label" in decorations[t]:
                viz.node(str(id(t)), decorations[t]["label"], style='filled', fillcolor=decorations[t]["color"],
                         fontsize=font_size)
            else:
                viz.node(str(id(t)), "", style='filled', fillcolor="black", fontsize=font_size)

    # places
    # add places, in order by their (unique) name, to avoid undeterminism in the visualization
    places_sort_list_im = sorted([x for x in list(net.places) if x in initial_marking], key=lambda x: x.name)
    places_sort_list_fm = sorted([x for x in list(net.places) if x in final_marking and not x in initial_marking],
                                 key=lambda x: x.name)
    places_sort_list_not_im_fm = sorted(
        [x for x in list(net.places) if x not in initial_marking and x not in final_marking], key=lambda x: x.name)
    # making the addition happen in this order:
    # - first, the places belonging to the initial marking
    # - after, the places not belonging neither to the initial marking and the final marking
    # - at last, the places belonging to the final marking (but not to the initial marking)
    # in this way, is more probable that the initial marking is on the left and the final on the right
    places_sort_list = places_sort_list_im + places_sort_list_not_im_fm + places_sort_list_fm

    for p in places_sort_list:
        if p in initial_marking:
            viz.node(str(id(p)), str(initial_marking[p]), style='filled', fillcolor="green", fontsize=font_size,
                     shape='circle', fixedsize='true', width='0.75')
        elif p in final_marking:
            viz.node(str(id(p)), "", style='filled', fillcolor="orange", fontsize=font_size, shape='circle',
                     fixedsize='true', width='0.75')
        else:
            if debug:
                viz.node(str(id(p)), str(p.name), fontsize=font_size, shape="ellipse")
            else:
                if p in decorations and "color" in decorations[p] and "label" in decorations[p]:
                    viz.node(str(id(p)), decorations[p]["label"], style='filled', fillcolor=decorations[p]["color"],
                             fontsize=font_size, shape="ellipse")
                else:
                    viz.node(str(id(p)), "", shape='circle', fixedsize='true', width='0.75')

    # add arcs, in order by their source and target objects names, to avoid undeterminism in the visualization
    arcs_sort_list = sorted(list(net.arcs), key=lambda x: (x.source.name, x.target.name))
    for a in arcs_sort_list:
        if a in decorations and "label" in decorations[a] and "penwidth" in decorations[a]:
            viz.edge(str(id(a.source)), str(id(a.target)), label=decorations[a]["label"],
                     penwidth=decorations[a]["penwidth"], fontsize=font_size)
        elif a in decorations and "color" in decorations[a]:
            viz.edge(str(id(a.source)), str(id(a.target)), color=decorations[a]["color"], fontsize=font_size)
        else:
            viz.edge(str(id(a.source)), str(id(a.target)), fontsize=font_size)
    viz.attr(overlap='false')

    viz.format = image_format

    return viz
