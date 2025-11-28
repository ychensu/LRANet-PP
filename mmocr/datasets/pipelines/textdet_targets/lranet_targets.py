import Polygon
import cv2
import numpy as np
from numpy.linalg import norm
import mmocr.utils.check_argument as check_argument
from .textsnake_targets import TextSnakeTargets
from scipy.interpolate import splprep, splev
from scipy.stats import multivariate_normal
PI = 3.1415926
from mmocr.utils.tps_util import TPS
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES



@PIPELINES.register_module()
class LRATargets(TextSnakeTargets):

    def __init__(self,
                 num_coefficients=14,
                 resample_step=4.0,
                 center_region_shrink_ratio=0.3,
                 level_size_divisors=(8, 16, 32),
                 level_proportion_range=((0, 0.4), (0.3, 0.7), (0.6, 1.0)),
                 tps_size=(1,1), #h,w
                 with_area=True,
                 interp=True,
                 gauss_center = False,
                 reoder=False,
                 shrink_ratio=0.4,
                 num_samples = 3,
                 ):

        super().__init__()
        assert isinstance(level_size_divisors, tuple)
        assert isinstance(level_proportion_range, tuple)
        assert len(level_size_divisors) == len(level_proportion_range)
        self.with_area = with_area
        self.num_coefficients = num_coefficients
        self.resample_step = resample_step
        self.tps_size = tps_size
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.level_size_divisors = level_size_divisors
        self.level_proportion_range = level_proportion_range
        self.shrink_ratio = shrink_ratio
        self.TPSGenerator = TPS(14, tps_size, fiducial_dist="edge")
        self.gauss_center = gauss_center
        self.reoder = reoder
        self.interp = interp
        self.num_samples = num_samples
        if self.gauss_center:
            clf = multivariate_normal(mean=[0, 0],
                                      cov=[[0.6, 0], [0, 0.6]]
                                      )
            div = 200
            # mask = np.zeros((div, div), np.float32)
            XX, YY = np.meshgrid(np.linspace(-2, 2, div), np.linspace(-2, 2, div ))
            Z = clf.pdf(np.dstack([XX, YY])).reshape(div, div)
            mask = Z / Z.max()
            self.gauss_mask = mask


    def generate_center_region_mask(self, img_size, text_polys):
        """Generate text center region mask.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            center_region_mask (ndarray): The text center region mask.
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size

        center_region_mask = np.zeros((h, w), np.uint8)

        center_region_boxes = []
        for poly in text_polys:
            assert len(poly) == 1
            polygon_points = poly[0].reshape(-1, 2)
            _, _, top_line, bot_line = self.reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self.resample_sidelines(
                top_line, bot_line, self.resample_step)
            # resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            line_head_shrink_len = norm(resampled_top_line[0] -
                                        resampled_bot_line[0]) / 4.0
            line_tail_shrink_len = norm(resampled_top_line[-1] -
                                        resampled_bot_line[-1]) / 4.0
            head_shrink_num = int(line_head_shrink_len // self.resample_step)
            tail_shrink_num = int(line_tail_shrink_len // self.resample_step)
            if len(center_line) > head_shrink_num + tail_shrink_num + 2:
                center_line = center_line[head_shrink_num:len(center_line) -
                                          tail_shrink_num]
                resampled_top_line = resampled_top_line[
                    head_shrink_num:len(resampled_top_line) - tail_shrink_num]
                resampled_bot_line = resampled_bot_line[
                    head_shrink_num:len(resampled_bot_line) - tail_shrink_num]

            for i in range(0, len(center_line) - 1):
                tl = center_line[i] + (resampled_top_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                tr = center_line[i + 1] + (
                    resampled_top_line[i + 1] -
                    center_line[i + 1]) * self.center_region_shrink_ratio
                br = center_line[i + 1] + (
                    resampled_bot_line[i + 1] -
                    center_line[i + 1]) * self.center_region_shrink_ratio
                bl = center_line[i] + (resampled_bot_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                current_center_box = np.vstack([tl, tr, br,
                                                bl]).astype(np.int32)
                center_region_boxes.append(current_center_box)

        cv2.fillPoly(center_region_mask, center_region_boxes, 1)
        return center_region_mask

    def generate_gauss_center_region_mask(self, img_size, text_polys,size_divisor):
        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size

        center_region_mask = np.zeros((h, w), np.float32)
        for poly in text_polys:
            gauss_center_mask = self.generate_gauss(img_size, poly, size_divisor)
            gauss_center_mask = gauss_center_mask / (gauss_center_mask.max() + 1e-5)
            col_mask = center_region_mask * gauss_center_mask > 0
            center_region_mask += gauss_center_mask/(gauss_center_mask.max()+1e-5)
            if col_mask.max()>0:
                center_region_mask[col_mask] = 0
        if center_region_mask.max() > 1:
            print(center_region_mask)
        return center_region_mask

    def generate_gauss(self, img_size, text_poly, size_divisor):
        head_edge, tail_edge, top_sideline, bot_sideline = self.reorder_poly_edge(text_poly[0].reshape(-1,2))
        resample_top_line, resample_bot_line = self.resample_polygon(top_sideline, bot_sideline)
        # clf = multivariate_normal(mean=[0, 0],
        #                           cov=[[0.9, 0], [0, 0.15]]
        #                           )
        div = 200
        h, w = img_size
        # mask[:div//4,:] = Z/Z.max()
        mask = self.gauss_mask.copy()
        tps = cv2.createThinPlateSplineShapeTransformer()
        sourcepts = np.concatenate([resample_top_line, resample_bot_line]).reshape(1,-1,2)
        sourcepts[:,:,0] *= div/w
        sourcepts[:,:,1] *= div/h
        matches = []
        for i in range(1, sourcepts.shape[1]+1):
            matches.append(cv2.DMatch(i, i, 0))
        targetpts = self.get_p(resample_top_line, resample_bot_line, div,div).reshape(1,-1,2)
        tps.estimateTransformation(sourcepts, targetpts, matches)
        img = tps.warpImage(mask)
        img = cv2.resize(img, (h,w))
        return img

    def get_p(self, top_line, bot_line, h,w):
        top_l = np.cumsum(norm(top_line[1:] - top_line[:-1], 2, axis=-1))
        top_l = np.insert(top_l, 0 ,0)
        top_y = np.ones(len(top_line)) * 0
        top_x = (top_l/top_l[-1]) * (w-1)
        bot_l = np.cumsum(norm(bot_line[1:] - bot_line[:-1], 2, axis=-1))
        bot_l = np.insert(bot_l, 0, 0)
        bot_x = (bot_l / bot_l[-1] ) * (w-1)
        bot_y = np.ones(len(top_line)) * (h-1)

        top_p = np.stack((top_x, top_y), -1)
        bot_p = np.stack((bot_x, bot_y), -1)

        p = np.concatenate((top_p, bot_p), 0)
        return p

    def resample_polygon(self, top_line,bot_line, n=None):
        """Resample one polygon with n points on its boundary.

        Args:
            polygon (list[float]): The input polygon.
            n (int): The number of resampled points.
        Returns:
            resampled_polygon (list[float]): The resampled polygon.
        """
        if n is None:
            n = self.num_coefficients // 2
        resample_line = []
        for polygon in [top_line, bot_line]:
            if polygon.shape[0] >= 3 and self.interp:
                x,y = polygon[:,0], polygon[:,1]
                tck, u = splprep([x, y], k=3 if polygon.shape[0] >=5 else 2, s=0)
                u = np.linspace(0, 1, num=n, endpoint=True)
                out = splev(u, tck)
                new_polygon = np.stack(out, axis=1).astype('float32')
            else:
                new_polygon = self.resample_line(polygon, n-1)
            resample_line.append(np.array(new_polygon))



        return resample_line # top line, bot line

    def normalize_polygon(self, polygon):

        temp_polygon = polygon - polygon.mean(axis=0)

        return temp_polygon/32

    def poly2T(self, polygon):
        """Convert polygon to tps cofficients

        Args:
            polygon (ndarray): An input polygon.
            center_point (tuple(int, int)): centerpoint of default box.
            side (float): side length of default box
        Returns:
            c (ndarray): Tps coefficients.
        """
        C_prime = polygon.reshape((1,-1,2))
        T = self.TPSGenerator.solve_T(C_prime)
        return T

    def poly2rotate_rect(self,polygon):
        rect = cv2.minAreaRect(polygon)
        box = cv2.boxPoints(rect)
        return box

    def clockwise(self, head_edge, tail_edge, top_sideline, bot_sideline):
        hc = head_edge.mean(axis=0)
        tc = tail_edge.mean(axis=0)
        d = (((hc - tc) ** 2).sum()) ** 0.5 + 0.1
        dx = np.abs(hc[0] - tc[0])
        if not dx / d <= 1:
            print(dx / d)
        angle = np.arccos(dx / d)
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


    def cal_tps_signature(self, top_line,bot_line):

        resample_top_line,resample_bot_line = self.resample_polygon(top_line,bot_line)
        resampled_polygon = np.concatenate([resample_top_line, resample_bot_line])
        resampled_polygon2 = np.concatenate([resample_top_line, resample_bot_line[::-1]])        
        tps_coeff = self.poly2T(resampled_polygon)

        return resampled_polygon2, tps_coeff.view(-1,1)

    def reorder_poly_edge(self, points):
        """Get the respective points composing head edge, tail edge, top
        sideline and bottom sideline.

        Args:
            points (ndarray): The points composing a text polygon.

        Returns:
            head_edge (ndarray): The two points composing the head edge of text
                polygon.
            tail_edge (ndarray): The two points composing the tail edge of text
                polygon.
            top_sideline (ndarray): The points composing top curved sideline of
                text polygon.
            bot_sideline (ndarray): The points composing bottom curved sideline
                of text polygon.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2
        # assert points.shape[0] % 2 == 0, "The data number should be 2 times"
        if self.reoder:
            head_edge, tail_edge, top_sideline, bot_sideline = super(TPSTargets, self).reorder_poly_edge(points)
            # bot_sideline = bot_sideline[::-1]
            top_sideline, bot_sideline,_ = self.clockwise(head_edge, tail_edge, top_sideline, bot_sideline)
        else:
            lh = points.shape[0]
            lhc2 = int(lh / 2)
            top_sideline = points[:lhc2]
            bot_sideline = points[lhc2:][::-1]
        head_edge = np.stack((top_sideline[0], bot_sideline[0]), 0)
        tail_edge = np.stack((top_sideline[-1], bot_sideline[-1]), 0)
        return head_edge, tail_edge, top_sideline, bot_sideline

    def generate_tps_maps(self, img_size, text_polys,text_polys_idx=None, img=None, level_size=None):

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        coeff_maps = np.zeros((28, h, w), dtype=np.float32)
        tps_coeffs = []
        for poly,poly_idx in zip(text_polys, text_polys_idx):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            mask = np.zeros((h, w), dtype=np.uint8)
            polygon = np.array(text_instance).reshape((1, -1, 2))
            head_edge, tail_edge, top_sideline, bot_sideline = self.reorder_poly_edge(polygon[0])
            cv2.fillPoly(mask, np.round(polygon).astype(np.int32), 1)
            resampled_polygon, tps_coeff = self.cal_tps_signature(top_sideline, bot_sideline)
            tps_coeffs.append(np.insert(tps_coeff.view(-1),0,poly_idx))


            yx = np.argwhere(mask > 0.5)
            y, x = yx[:, 0], yx[:, 1]
            batch_T = np.zeros((h, w, 14, 2))
            batch_T[y,x,:,:] = resampled_polygon
            batch_T = batch_T.reshape(h, w, -1).transpose(2, 0, 1)
            coeff_maps[:, y,x] = batch_T[:,y,x]

        if len(tps_coeffs) > 0:
            tps_coeffs = np.stack(tps_coeffs, 0)
        else:
            tps_coeffs = np.array([])
        return coeff_maps, tps_coeffs

    def generate_text_region_mask(self, img_size, text_polys, text_polys_idx):
        """Generate text center region mask and geometry attribute maps.

        Args:
            img_size (tuple): The image size (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            text_region_mask (ndarray): The text region mask.
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        text_region_mask = np.zeros((h, w), dtype=np.uint8)

        for poly, poly_idx in zip(text_polys, text_polys_idx):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            polygon = np.array(
                np.round(text_instance), dtype=np.int).reshape((1, -1, 2))
            if self.with_area:
                cv2.fillPoly(text_region_mask, polygon, poly_idx)
            else:
                cv2.fillPoly(text_region_mask, polygon, 1)
        return text_region_mask

    def generate_level_targets(self, img_size, text_polys, ignore_polys,img=None):
        """Generate ground truth target on each level.

        Args:
            img_size (list[int]): Shape of input image.
            text_polys (list[list[ndarray]]): A list of ground truth polygons.
            ignore_polys (list[list[ndarray]]): A list of ignored polygons.
        Returns:
            level_maps (list(ndarray)): A list of ground target on each level.
            :param img:
        """
        h, w = img_size
        lv_size_divs = self.level_size_divisors
        lv_proportion_range = self.level_proportion_range
        lv_text_polys = [[] for i in range(len(lv_size_divs))]
        lv_text_polys_idx = [[] for i in range(len(lv_size_divs))]
        lv_ignore_polys = [[] for i in range(len(lv_size_divs))]
        polygons_area = []
        zeros_mask = np.zeros((h,w), dtype=np.float32)
        level_maps = []
        lv_tps_coeffs = [[] for i in range(len(lv_size_divs))]
        for poly_idx, poly in enumerate(text_polys):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int).reshape((1, -1, 2))
            tl_x, tl_y, box_w, box_h = cv2.boundingRect(polygon)
            # assert box_w <= 200 or box_h <= 200, 'Box out of range'
            # max_l = max(box_h, box_w)

            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_text_polys[ind].append([poly[0] / lv_size_divs[ind]])
                    lv_text_polys_idx[ind].append(poly_idx+1)

            if self.with_area:
                polygon_area = Polygon.Polygon(poly[0].reshape(-1,2)).area()
                polygons_area.append(polygon_area)


        for ignore_poly in ignore_polys:
            assert len(ignore_poly) == 1
            text_instance = [[ignore_poly[0][i], ignore_poly[0][i + 1]]
                             for i in range(0, len(ignore_poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int).reshape((1, -1, 2))
            _, _, box_w, box_h = cv2.boundingRect(polygon)
            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_ignore_polys[ind].append(
                        [ignore_poly[0] / lv_size_divs[ind]])

        for ind, size_divisor in enumerate(lv_size_divs):
            current_level_maps = []
            level_img_size = (h // size_divisor, w // size_divisor)
            text_region = self.generate_text_region_mask(
                level_img_size, lv_text_polys[ind], lv_text_polys_idx[ind])[None]
            current_level_maps.append(text_region)

            if self.gauss_center:
                center_region = self.generate_gauss_center_region_mask(
                    level_img_size, lv_text_polys[ind], size_divisor)[None]
            else:
                center_region = self.generate_center_region_mask(
                    level_img_size, lv_text_polys[ind])[None]
            current_level_maps.append(center_region)

            effective_mask = self.generate_effective_mask(
                level_img_size, lv_ignore_polys[ind])[None]
            current_level_maps.append(effective_mask)

            coeff_maps,  tps_coeffs = self.generate_tps_maps(
                level_img_size, lv_text_polys[ind],lv_text_polys_idx[ind])

            current_level_maps.append(coeff_maps)
            lv_tps_coeffs[ind] = tps_coeffs

            level_maps.append(np.concatenate(current_level_maps))


        resampled_polys = []  
        for j in range(len(text_polys)):
            polygon = text_polys[j][0].reshape(-1, 2)
            _, _, top_sideline, bot_sideline = self.reorder_poly_edge(polygon)
            resample_top_line, resample_bot_line = self.resample_polygon(top_sideline, bot_sideline)
            flat_contour = np.concatenate([resample_top_line, resample_bot_line[::-1]]).flatten()
            flat_contour = np.expand_dims(flat_contour, axis=1)
            resampled_polys.append(flat_contour.flatten())           
        resampled_polys = np.array(resampled_polys)


        if resampled_polys.shape[0] > 0:
            resampled_polys = np.concatenate([resampled_polys] * self.num_samples, axis=0)

        if self.with_area and len(polygons_area) > 0:
            polygons_area = np.array(polygons_area)
        else:
            polygons_area = np.array([])

        lv_text_polys_idx = [np.array(l) for l in lv_text_polys_idx]

        return level_maps, lv_text_polys_idx,  polygons_area, lv_tps_coeffs, resampled_polys 

    def generate_targets(self, results):
        """Generate the ground truth targets for FCENet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """

        assert isinstance(results, dict)

        polygon_masks = results['gt_masks'].masks
        polygon_masks_ignore = results['gt_masks_ignore'].masks
        gt_texts = results['texts']
        h, w, _ = results['img_shape']

        level_maps, lv_text_polys_idx, polygons_area, lv_tps_coeffs, resampled_polys = self.generate_level_targets((h, w), polygon_masks,
                                                 polygon_masks_ignore, results['img'])

        results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        mapping = {
            'p3_maps': level_maps[0],
            'p4_maps': level_maps[1],
            'p5_maps': level_maps[2],
            'lv_text_polys_idx':lv_text_polys_idx,
            'polygons_area': polygons_area,
            'gt_texts': DC(gt_texts, cpu_only=True),
            'lv_tps_coeffs': lv_tps_coeffs,
            'lra_polys': resampled_polys
        }
        if len(self.level_size_divisors) == 4:
            mapping['p6_maps'] = level_maps[3]
        for key, value in mapping.items():
            results[key] = value

        return results
