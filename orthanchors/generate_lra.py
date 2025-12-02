import os
import json
import torch
import argparse
import numpy as np
import torch.linalg as LA
from numpy.linalg import norm
from scipy.interpolate import splprep, splev
from sklearn.utils.extmath import randomized_svd


class FMS:
    """Fast Median Subspace solver

    The optimization objective is the following problem:
        min_{V} ||(I-VV^T) X||_{1,2}  s.t.  V^T V = I
    where:
        V : optimization variable with shape [n_features, n_directions],
            and is constrained to have orthonormal columns
        X : data matrix with shape [n_features, n_samples]
        ||.||_{1,2} : mixed l1/l2 norm for any matrix A is defined by
            ||A||_{1,2} = \\sum_i ||row_i of A||_2

    We solve the problem described above by Fast Median Subspace (FMS) method, 
    which is proposed in the following paper:

    Lerman, G., & Maunu, T. (2018). Fast, robust and non-convex subspace recovery. 
    Information and Inference: A Journal of the IMA, 7(2), 277-336.

    Please refer to the paper for details, and kindly cite the work 
    if you find it is useful.

    Parameters
    ----------
    d : int, required
        The desired dimension of the underlying subspace
    max_iter : int, optional
        The maximum number of iterations
    eps : float, optional
        The safeguard parameter
    tol : float, optional
        The termination tolerance parameter
    no_random : bool, optional
        The parameter controls if randomized SVD is allowed
    spherical: bool, optional
        The parameter controls if normalize the data to unit sphere
    verbose: bool, optional
        The parameter controls if output info during running

    Attributes
    ----------
    V : tensor, shape [n_features, d]
        computed optimization variable in the problem formulation

    Examples
    --------
        See test.py

    Copyright (C) 2022 Tianyu Ding <tianyu.ding0@gmail.com>
    """

    def __init__(
        self,
        max_iter=1000,
        eps=1e-10,
        tol=1e-18,
        no_random=True,
        spherical=True,
        verbose=False,
    ):
        self.max_iter = max_iter
        self.eps = eps
        self.tol = tol
        self.no_random = no_random
        self.spherical = spherical
        self.verbose = verbose

    def run(self, X, d):
        '''
            X: N x D data set with N points dim D
            d: dim of the subspace to find
        '''
        D = X.shape[1]

        if not (0 < d < D):
            raise ValueError("The problem is not well-defined.")

        def loss(V):
            return torch.sum(torch.sqrt(torch.sum(((torch.eye(D)-V @ V.t()) @ X.t()) ** 2, dim=0)))

        if self.spherical:
            # spherize the data
            self.X = X / LA.norm(X, axis=1, keepdims=True)
        else:
            self.X = X

        mn = min(X.shape)

        dist = torch.FloatTensor([1e5])
        iter = 0

        if d > 0.6 * mn or self.no_random is True:
            _, _, Vh = LA.svd(self.X, full_matrices=False)
            Vi = Vh.t()
            Vi = Vi[:, :d]  # D x d
        else:
            Vi, _, _ = randomized_svd(
                self.X.t().numpy(), n_components=d, random_state=0)
            Vi = torch.from_numpy(Vi)

        Vi_prev = Vi
        while dist > self.tol and iter < self.max_iter:

            if self.verbose:
                print('Iter: %3d, L1 loss: %10.3f, dist: %10g'
                      % (iter, loss(Vi_prev).item(), dist.item()))

            # project datapoints onto the orthogonal complement
            C = self.X.t() - Vi @ (Vi.t() @ self.X.t())  # D x N

            scale = LA.norm(C, axis=0, keepdims=True)  # 1 x N

            Y = self.X * torch.min(scale.t()**(-.5), torch.tensor(1/self.eps))

            if d > 0.6 * mn or self.no_random is True:
                _, _, Vh = LA.svd(Y, full_matrices=False)
                Vi = Vh.t()
                Vi = Vi[:, :d]  # D x d
            else:
                Vi, _, _ = randomized_svd(
                    Y.t().numpy(), n_components=d, random_state=0)
                Vi = torch.from_numpy(Vi)

            dist = self.comp_dist(Vi, Vi_prev)

            Vi_prev = Vi
            iter += 1

        return Vi_prev

    def comp_dist(self, S1, S2):
        A = S1.t() @ S2
        U, _, Vh = LA.svd(A)
        Q = U @ Vh
        dist = LA.norm(S2-S1@Q, 'fro') / torch.sqrt(torch.tensor(S1.shape[1]))
        return dist



def resample_line(line, n):
    """Resample n points on a line.

    Args:
        line (ndarray): The points composing a line.
        n (int): The resampled points number.

    Returns:
        resampled_line (ndarray): The points composing the resampled line.
    """

    assert line.ndim == 2
    assert line.shape[0] >= 2
    assert line.shape[1] == 2
    assert isinstance(n, int)
    assert n > 0

    length_list = [
        norm(line[i + 1] - line[i]) for i in range(len(line) - 1)
    ]
    total_length = sum(length_list)
    length_cumsum = np.cumsum([0.0] + length_list)
    delta_length = total_length / (float(n) + 1e-8)

    current_edge_ind = 0
    resampled_line = [line[0]]

    for i in range(1, n):
        current_line_len = i * delta_length

        while current_line_len >= length_cumsum[current_edge_ind + 1]:
            current_edge_ind += 1
        current_edge_end_shift = current_line_len - length_cumsum[
            current_edge_ind]
        end_shift_ratio = current_edge_end_shift / length_list[
            current_edge_ind]
        current_point = line[current_edge_ind] + (
            line[current_edge_ind + 1] -
            line[current_edge_ind]) * end_shift_ratio
        resampled_line.append(current_point)

    resampled_line.append(line[-1])
    resampled_line = np.array(resampled_line)

    return resampled_line


def resample_polygon(top_line,bot_line, n=7):

    resample_line = []
    for polygon in [top_line, bot_line]:
        if polygon.shape[0] >= 3:
            x,y = polygon[:,0], polygon[:,1]
            tck, u = splprep([x, y], k=3 if polygon.shape[0] >=5 else 2, s=0)
            u = np.linspace(0, 1, num=n, endpoint=True)
            out = splev(u, tck)
            new_polygon = np.stack(out, axis=1).astype('float32')
        else:
            new_polygon = resample_line(polygon, n-1)

        resample_line.append(np.array(new_polygon))

    return resample_line # top line, bot line



def vector_angle(vec1, vec2):
    if vec1.ndim > 1:
        unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8).reshape((-1, 1))
    else:
        unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8)
    if vec2.ndim > 1:
        unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8).reshape((-1, 1))
    else:
        unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8)
    return np.arccos(
        np.clip(np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))

def vector_slope(vec):
    assert len(vec) == 2
    return abs(vec[1] / (vec[0] + 1e-8))

def vector_sin(vec):
    assert len(vec) == 2
    return vec[1] / (norm(vec) + 1e-8)

def vector_cos(vec):
    assert len(vec) == 2
    return vec[0] / (norm(vec) + 1e-8)

def find_head_tail(points, orientation_thr):

    assert points.ndim == 2
    assert points.shape[0] >= 4
    assert points.shape[1] == 2
    assert isinstance(orientation_thr, float)

    if len(points) > 4:
        pad_points = np.vstack([points, points[0]])
        edge_vec = pad_points[1:] - pad_points[:-1]

        theta_sum = []
        adjacent_vec_theta = []
        for i, edge_vec1 in enumerate(edge_vec):
            adjacent_ind = [x % len(edge_vec) for x in [i - 1, i + 1]]
            adjacent_edge_vec = edge_vec[adjacent_ind]
            temp_theta_sum = np.sum(
                vector_angle(edge_vec1, adjacent_edge_vec))
            temp_adjacent_theta = vector_angle(
                adjacent_edge_vec[0], adjacent_edge_vec[1])
            theta_sum.append(temp_theta_sum)
            adjacent_vec_theta.append(temp_adjacent_theta)
        theta_sum_score = np.array(theta_sum) / np.pi
        adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
        poly_center = np.mean(points, axis=0)
        edge_dist = np.maximum(
            norm(pad_points[1:] - poly_center, axis=-1),
            norm(pad_points[:-1] - poly_center, axis=-1))
        dist_score = edge_dist / np.max(edge_dist)
        position_score = np.zeros(len(edge_vec))
        score = 0.5 * theta_sum_score + 0.15 * adjacent_theta_score
        score += 0.35 * dist_score
        if len(points) % 2 == 0:
            position_score[(len(score) // 2 - 1)] += 1
            position_score[-1] += 1
        score += 0.1 * position_score
        pad_score = np.concatenate([score, score])
        score_matrix = np.zeros((len(score), len(score) - 3))
        x = np.arange(len(score) - 3) / float(len(score) - 4)
        gaussian = 1. / (np.sqrt(2. * np.pi) * 0.5) * np.exp(-np.power(
            (x - 0.5) / 0.5, 2.) / 2)
        gaussian = gaussian / np.max(gaussian)
        for i in range(len(score)):
            score_matrix[i, :] = score[i] + pad_score[
                (i + 2):(i + len(score) - 1)] * gaussian * 0.3

        head_start, tail_increment = np.unravel_index(
            score_matrix.argmax(), score_matrix.shape)
        tail_start = (head_start + tail_increment + 2) % len(points)
        head_end = (head_start + 1) % len(points)
        tail_end = (tail_start + 1) % len(points)

        if head_end > tail_end:
            head_start, tail_start = tail_start, head_start
            head_end, tail_end = tail_end, head_end
        head_inds = [head_start, head_end]
        tail_inds = [tail_start, tail_end]
    else:
        if vector_slope(points[1] - points[0]) + vector_slope(
                points[3] - points[2]) < vector_slope(
                    points[2] - points[1]) + vector_slope(points[0] -
                                                                points[3]):
            horizontal_edge_inds = [[0, 1], [2, 3]]
            vertical_edge_inds = [[3, 0], [1, 2]]
        else:
            horizontal_edge_inds = [[3, 0], [1, 2]]
            vertical_edge_inds = [[0, 1], [2, 3]]

        vertical_len_sum = norm(points[vertical_edge_inds[0][0]] -
                                points[vertical_edge_inds[0][1]]) + norm(
                                    points[vertical_edge_inds[1][0]] -
                                    points[vertical_edge_inds[1][1]])
        horizontal_len_sum = norm(
            points[horizontal_edge_inds[0][0]] -
            points[horizontal_edge_inds[0][1]]) + norm(
                points[horizontal_edge_inds[1][0]] -
                points[horizontal_edge_inds[1][1]])

        if vertical_len_sum > horizontal_len_sum * orientation_thr:
            head_inds = horizontal_edge_inds[0]
            tail_inds = horizontal_edge_inds[1]
        else:
            head_inds = vertical_edge_inds[0]
            tail_inds = vertical_edge_inds[1]

    return head_inds, tail_inds


def clockwise(head_edge, tail_edge, top_sideline, bot_sideline):
    hc = head_edge.mean(axis=0)
    tc = tail_edge.mean(axis=0)
    d = (((hc - tc) ** 2).sum()) ** 0.5 + 0.1
    dx = np.abs(hc[0] - tc[0])
    if not dx / d <= 1:
        print(dx / d)
    angle = np.arccos(dx / d)
    PI = 3.1415926
    direction = 0 if angle <= PI / 4 else 1  # 0 horizontal, 1 vertical
    if top_sideline[0, direction] > top_sideline[-1, direction]:
        top_indx = np.arange(top_sideline.shape[0] - 1, -1, -1)
    else:
        top_indx = np.arange(0, top_sideline.shape[0])
    top_sideline = top_sideline[top_indx]
    if direction == 1 and top_sideline[0, direction] < top_sideline[-1, direction]:
        top_indx = np.arange(top_sideline.shape[0] - 1, -1, -1)
        top_sideline = top_sideline[top_indx]

    if bot_sideline[0, direction] > bot_sideline[-1, direction]:
        bot_indx = np.arange(bot_sideline.shape[0] - 1, -1, -1)
    else:
        bot_indx = np.arange(0, bot_sideline.shape[0])
    bot_sideline = bot_sideline[bot_indx]
    if direction == 1 and bot_sideline[0, direction] < bot_sideline[-1, direction]:
        bot_indx = np.arange(bot_sideline.shape[0] - 1, -1, -1)
        bot_sideline = bot_sideline[bot_indx]
    if top_sideline[:, 1 - direction].mean() > bot_sideline[:, 1 - direction].mean():
        top_sideline, bot_sideline = bot_sideline, top_sideline

    return top_sideline, bot_sideline, direction
    

def reorder_poly_edge(points):

    assert points.ndim == 2
    assert points.shape[0] >= 4
    assert points.shape[1] == 2

    orientation_thr=2.0
    head_inds, tail_inds = find_head_tail(points,
                                                orientation_thr)
    head_edge, tail_edge = points[head_inds], points[tail_inds]

    pad_points = np.vstack([points, points])
    if tail_inds[1] < 1:
        tail_inds[1] = len(points)
    sideline1 = pad_points[head_inds[1]:tail_inds[1]]
    sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
    sideline_mean_shift = np.mean(
        sideline1, axis=0) - np.mean(
            sideline2, axis=0)

    if sideline_mean_shift[1] > 0:
        top_sideline, bot_sideline = sideline2, sideline1
    else:
        top_sideline, bot_sideline = sideline1, sideline2
        
    top_sideline, bot_sideline,_ = clockwise(head_edge, tail_edge, top_sideline, bot_sideline)
    bot_sideline = bot_sideline[::-1]
    return top_sideline, bot_sideline


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run FMS to generate Low-Rank Approximation (LRA) basis vectors.")
    
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to the dataset annotation JSON file')
    parser.add_argument('--output_dir', type=str, default="./",
                        help='Directory to save the output .npz file')
    parser.add_argument('--n_components', type=int, default=14,
                        help='Dimensionality of the subspace (LRA dimension)')

    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: Dataset file not found at {args.json_path}")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading annotations from: {args.json_path}")
    with open(args.json_path, 'r') as f:
        json_data = json.load(f)
    
    annotations = json_data['annotations']
    resampled_lines_list = []

    print(f"Processing {len(annotations)} text instances...")

    for i, annotation_item in enumerate(annotations):
        # Extract segmentation points
        seg_points = annotation_item['segmentation'][0]
        polygon = np.array(seg_points).reshape(-1, 2).astype(np.float32)
        
        # Filter out duplicate points to prevent errors in spline interpolation (splprep)
        _, idx = np.unique(polygon, axis=0, return_index=True)
        polygon = polygon[np.sort(idx)]

        # Filter invalid polygons
        if polygon.shape[0] < 4:
            continue

        # Resample contours to fixed number of points
        top_sideline, bot_sideline = reorder_poly_edge(polygon)
        resampled_top, resampled_bot = resample_polygon(top_sideline, bot_sideline)
        
        resampled_line = np.concatenate([resampled_top, resampled_bot], axis=0).flatten()
        resampled_lines_list.append(resampled_line)

    contour_matrix_np = np.array(resampled_lines_list)
    print(f"Constructed data matrix with shape: {contour_matrix_np.shape}")

    # --- Run FMS ---
    print(f"Running FMS to recover {args.n_components} basis vectors...")
    fms_solver = FMS(verbose=True)
    basis_vectors_tensor = fms_solver.run(torch.from_numpy(contour_matrix_np), args.n_components)

    # --- Save Results ---
    output_filename = os.path.join(args.output_dir, f'fms{args.n_components}.npz')
    print(f"Saving LRA encoding matrix to: {output_filename}")
    np.savez(
        output_filename,
        components_c=basis_vectors_tensor.clone().permute(1, 0).numpy()
    )
    print("Done.")