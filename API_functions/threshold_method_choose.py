from . import threshold_position_independent as mpi
from . import threshold_position_dependent as mpd
from . import pre_process as mpp


def seg_method_choose(text):
    method = {
        'origin': (mpi.origin, True),
        'kmeans': (mpi.kmeans_3d, True),
        'gmm': (mpi.gmm_3d, True),
        'otsu': (mpi.otsu_3d, True),
        'kapur_entropy': (mpi.kapur_entropy_3d, True),
        'watershed': (mpd.watershed, False)
    }
    return method.get(text, None)    # return None if not found


def pre_method_choose(text):
    methods = {
        'origin': (mpp.origin, {}),
        'gamma': (mpp.adjust_gamma, {'gamma_value': 1.0}),  # default gamma value is 1.0
        'equalized': (mpp.equalized_hist, {}),
        'median': (mpp.median, {})
    }

    return methods.get(text, (None, {}))    # return None if not found
