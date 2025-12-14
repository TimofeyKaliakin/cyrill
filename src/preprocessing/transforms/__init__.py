from .scale import ScaleAugmentation
from .shear import ShearAugmentation
from .erosion import ErosionAugmentation
from .dilation import DilationAugmentation
from .griddistortion import GridDistortionAugmentation
from .motion_blur import MotionBlurAugmentation
from .elastic_transform import ElasticTransformAugmentation
from .bad_photo_copy import BadPhotoCopyAugmentation

__all__ = ['ScaleAugmentation',  'ShearAugmentation', 'ErosionAugmentation',
           'DilationAugmentation', 'GridDistortionAugmentation',
           'MotionBlurAugmentation', 'ElasticTransformAugmentation',
           'BadPhotoCopyAugmentation']
