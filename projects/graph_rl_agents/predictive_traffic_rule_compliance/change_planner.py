import numpy as np
from shapely.geometry.point import Point

def dense_cv(cv, interval=0.1):
    """
    将车道线的点加密为每隔 interval 米一个点。

    :param cv: 原始车道线点序列，格式为[[x1, y1], [x2, y2], ...]
    :param interval: 生成点的间隔，默认为0.1米
    :return: 加密后的点序列
    """
    new_cv = [cv[0]]  # 初始化加密后的点序列，包含起点
    for i in range(1, len(cv)):
        start_point = np.array(cv[i - 1])
        end_point = np.array(cv[i])
        segment_length = np.linalg.norm(end_point - start_point)

        # 在当前段上插入新的点
        num_points = int(segment_length // interval)
        direction = (end_point - start_point) / segment_length
        for j in range(1, num_points + 1):
            new_point = start_point + direction * interval * j
            new_cv.append(new_point.tolist())

    new_cv.append(cv[-1])  # 最后一个点为终点
    return np.array(new_cv)


def compute_arc_length(center_line):
    """
    根据中心线点计算每个点的累计弧长距离。
    :param center_line: 中心线坐标点集，形状 (n, 2)。
    :return: 每个点的累计距离列表 (弧长)。
    """
    arc_lengths = [0.0]  # 第一个点的累积距离为 0
    for i in range(1, len(center_line)):
        segment_length = np.linalg.norm(center_line[i] - center_line[i - 1])
        arc_lengths.append(arc_lengths[-1] + segment_length)
    return np.array(arc_lengths)


def distance_lanelet(center_line, center_line_s, p1, p2):
    """
    计算点 p1 和点 p2 在道路中心线上的距离
    center_line : 道路中心线坐标点集 (2D array, shape: (n, 2))
    center_line_s : 累计距离 (弧长)
    p1, p2 : 要计算距离的两个点 (形状应为 1D array, e.g., [x, y])
    """
    # 确保 center_line 是 numpy 数组
    center_line = np.array(center_line)

    if isinstance(p1, Point):
        p1 = np.array([p1.x, p1.y])
    if isinstance(p2, Point):
        p2 = np.array([p2.x, p2.y])

    # 转置 center_line 确保其为 (n, 2) 形状，而不是 (2, n)
    if center_line.shape[0] == 2 and center_line.shape[1] > 2:
        center_line = center_line.T  # 转置，使其形状为 (n, 2)

    # 确保 p1 和 p2 是二维数组，能够与 center_line 进行广播
    p1 = p1.reshape(1, -1)  # p1 变为 (1, 2)
    p2 = p2.reshape(1, -1)  # p2 变为 (1, 2)

    # 计算 p1 和 p2 到每个中心线点的距离
    d1 = np.linalg.norm(center_line - p1, axis=1)  # p1 到每个中心线点的距离
    d2 = np.linalg.norm(center_line - p2, axis=1)  # p2 到每个中心线点的距离

    idx1 = np.argmin(d1)  # p1 在中心线上的最近点的索引
    idx2 = np.argmin(d2)  # p2 在中心线上的最近点的索引

    # 返回 p1 和 p2 在中心线上的累计距离差
    return abs(center_line_s[idx2] - center_line_s[idx1])

def check_state(scenario, ego_state, route):
    """check if ego car is straight-going /incoming /in-intersection"""
    lanelet_ego = None
    lanelet_id_ego = lanelet_ego
    ln = scenario.lanelet_network
    # find current lanelet
    potential_ego_lanelet_id_list = \
        scenario.lanelet_network.find_lanelet_by_position([ego_state.position])[0]
    for idx in potential_ego_lanelet_id_list:
        if idx in route.lanelet_ids:
            lanelet_id_ego = idx
    lanelet_ego = lanelet_id_ego
    print('current lanelet id:', lanelet_ego)

    for idx_inter, intersection in enumerate(ln.intersections):
        incomings = intersection.incomings

        for idx_inc, incoming in enumerate(incomings):
            incoming_lanelets = list(incoming.incoming_lanelets)
            in_intersection_lanelets = list(incoming.successors_straight) + \
                                       list(incoming.successors_right) + list(incoming.successors_left)

            for laneletid in incoming_lanelets:
                if lanelet_ego == laneletid:
                    lanelet_state = 2  # incoming

            for laneletid in in_intersection_lanelets:
                if lanelet_ego == laneletid:
                    lanelet_state = 3  # in-intersection

    if lanelet_state is None:
        lanelet_state = 1  # straighting-going

    return lanelet_state