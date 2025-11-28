from .transforms import RandomCropPolyInstances
from mmdet.datasets.builder import PIPELINES
import numpy as np
from mmdet.core import BitmapMasks, PolygonMasks


@PIPELINES.register_module()
class RandomCropPolyInstancesWithText(RandomCropPolyInstances):

    def __call__(self, results):
        if len(results[self.instance_key].masks) < 1:
            return results
        if np.random.random_sample() < self.crop_ratio:
            res = results.copy()
            crop_box = self.sample_crop_box(results['img'].shape, results)
            results['crop_region'] = crop_box
            img = self.crop_img(results['img'], crop_box)
            results['img'] = img
            results['img_shape'] = img.shape

            # crop and filter masks
            x1, y1, x2, y2 = crop_box
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            labels = results['gt_labels']
            texts = results['texts']
            valid_labels = []
            valid_texts = []
            for key in results.get('mask_fields', []):
                if len(results[key].masks) == 0:
                    continue
                results[key] = results[key].crop(crop_box)
                # filter out polygons beyond crop box.
                masks = results[key].masks
                valid_masks_list = []

                for ind, mask in enumerate(masks):
                    assert len(mask) == 1
                    polygon = mask[0].reshape((-1, 2))
                    if (polygon[:, 0] >
                        -4).all() and (polygon[:, 0] < w + 4).all() and (
                            polygon[:, 1] > -4).all() and (polygon[:, 1] <
                                                           h + 4).all():
                        mask[0][::2] = np.clip(mask[0][::2], 0, w)
                        mask[0][1::2] = np.clip(mask[0][1::2], 0, h)
                        if key == self.instance_key:
                            valid_labels.append(labels[ind])
                            valid_texts.append(texts[ind])
                        valid_masks_list.append(mask)

                results[key] = PolygonMasks(valid_masks_list, h, w)
            results['gt_labels'] = np.array(valid_labels)
            results['texts'] = valid_texts
            if len(valid_labels)==0:
                print(results)

        return results
